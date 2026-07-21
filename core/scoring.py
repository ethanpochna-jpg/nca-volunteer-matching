"""Likert constants, prompt factory, scoring waves, tiering, and caps.

Phase 4 verbatim move out of app.py (SECTION 2 scoring constants and
SECTION 7B minus the graph node).  Zero behavior edits; cross-module
references are import-qualified only.
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from core import llm, matching

# ── Likert scoring constants (S2) ─────────────────────────────────────────
# Source of truth: plaintext_ranking_prompts.txt (verbatim — item wording
# changes are a spec change, not a refactor).  The model sees an item's
# text and the anchor labels ONLY; the tiering score collapse (T2B/Neutral/
# B2B → +3/+1/−1) happens in code and is never shown at scoring time.

LIKERT_ANCHORS = (
    ("Strongly agree", 5),
    ("Somewhat agree", 4),
    ("Neutral", 3),
    ("Somewhat disagree", 2),
    ("Strongly disagree", 1),
)

LIKERT_ITEMS = (
    {
        "key": "overall_fit",
        "text": (
            "Based on your understanding of this volunteer, the requested "
            "task, and the nature of volunteering in general, rate your "
            "agreement with the following statement:\n"
            "This person is a great fit for this role."
        ),
    },
    {
        "key": "schedule_friction",
        "text": (
            "Based on the data you have on this volunteer's schedule and "
            "the nature of the request, rate your agreement with the "
            "following statement:\n"
            "The timeline/schedule in the request, if specified, would "
            "cause no friction given this person's availability, schedule, "
            "or notification preferences."
        ),
    },
    {
        "key": "willingness",
        "text": (
            "Based on this volunteer's preferences, history, and "
            "characteristics, they will likely be glad to take this request."
        ),
    },
    {
        "key": "recommendation",
        "text": (
            "Based on what you know about this volunteer, the request, and "
            "volunteering in general, you would recommend this volunteer "
            "for this role."
        ),
    },
)

# Raw 1–5 selection → tiering score.  Top-two-box +3, Neutral +1,
# bottom-two-box −1; attainable four-item sums are the even values in
# [−4, 12].
SCORE_MAP = {5: 3, 4: 3, 3: 1, 2: -1, 1: -1}

# ── Tier thresholds and ordering (S4) ─────────────────────────────────────
# Thresholds live in CODE, never in prompts (I4).  Sum ≥ 10 ⟺ zero B2B and
# at most one Neutral; 2–8 → Good; ≤ 0 → Technical.  Almost Match is
# assigned upstream by the matcher, never by score.
PERFECT_MIN = 10
GOOD_MIN = 2

TIER_RANK = {
    "Perfect Match": 0,
    "Good Match": 1,
    "Technical Match": 2,
    "Almost Match": 3,
}


VOLUNTEERS_IN_FLIGHT = 4
_SCORER_MAX_WORKERS = VOLUNTEERS_IN_FLIGHT * len(LIKERT_ITEMS)

SCORING_UNAVAILABLE_NOTE = (
    "Automated scoring was unavailable for this volunteer — shown as "
    "Technical Match by policy (passes all hard requirements)."
)


def collapse_box(selection: int) -> str:
    """Raw 1–5 → box label.  Code-side only; the model never sees boxes."""
    return {5: "T2B", 4: "T2B", 3: "Neutral", 2: "B2B", 1: "B2B"}[selection]


def partition_waves(units: list, wave_size: int = VOLUNTEERS_IN_FLIGHT) -> list:
    """Chunk scoring units into fixed-size waves, preserving order."""
    return [units[i:i + wave_size] for i in range(0, len(units), wave_size)]


def build_scorer_shared_context(state: dict, need_set: dict) -> str:
    """Shared system context for every item call of one need set's group.

    Contents per PLAN S3: original request, need-set description, STATED
    soft preferences, unconfirmed suggested skills labeled as context only
    (the surviving intent of old audit fix 9), and the recurring line when
    set (fix 12's use point).  No capacity numbers, history, or home area
    — those are deterministic display concerns.
    """
    lines = [
        "You are scoring one volunteer's fit for a volunteer request at "
        "Northbridge Community Alliance.",
        "Rate your agreement with the statement presented, based only on "
        "the information provided.",
        "",
        "=== REQUEST ===",
        str(state.get("user_prompt", "")),
        "",
        "=== NEED SET ===",
        str(need_set.get("description", "")),
    ]

    soft = state.get("soft_preferences", "")
    lines += ["", "=== STATED SOFT PREFERENCES ==="]
    lines.append(soft if soft else "None stated.")

    unchecked = state.get("unchecked_skills", [])
    if unchecked:
        lines += [
            "",
            f"Suggested skills ({', '.join(unchecked)}) are suggested but "
            f"not required — context only.",
        ]

    if state.get("is_recurring"):
        end = state.get("recurring_end_date") or "an open-ended date"
        lines += [
            "",
            f"This is a recurring need through {end}; weigh sustained "
            f"availability.",
        ]

    return "\n".join(lines)


def build_volunteer_profile(vol_row) -> str:
    """Compressed volunteer profile — the same soft-fit fields the old
    recommender saw; capacity, history, and home area stay out."""
    return f"""VOLUNTEER: {vol_row['preferred_name']} ({vol_row['volunteer_id']})
  Skills: {matching.truncate_text(vol_row['skills'])}
  Preferred Roles: {matching.truncate_text(vol_row.get('preferred_roles', ''))}
  Certifications: {matching.truncate_text(vol_row['certifications'])}
  Availability: {vol_row['availability_days']} / {vol_row['availability_time_blocks']}
  Availability Notes: {matching.truncate_text(vol_row.get('availability_notes', ''))}
  Transportation: {vol_row.get('transportation', 'N/A')}
  Languages: {vol_row['languages']}
  Notes: {matching.truncate_text(vol_row.get('notes', ''))}"""


def build_item_prompt(item: dict) -> str:
    """One item's statement plus the anchor labels — nothing else."""
    anchor_lines = "\n".join(
        f"{value} = {label}" for label, value in LIKERT_ANCHORS
    )
    return f"{item['text']}\n\nAnchors:\n{anchor_lines}"


def _score_item_with_retry(shared_ctx: str, profile: str, item: dict) -> int:
    """One item call with exactly one jittered retry.

    The client itself runs max_retries=0, so this is the ONLY retry layer.
    A second failure propagates — the caller applies the volunteer-level
    Technical Match fallback (never a fake Neutral, which would skew the
    sum).
    """
    item_prompt = build_item_prompt(item)
    try:
        return llm.call_likert_item(shared_ctx, profile, item_prompt)
    except Exception:
        time.sleep(random.uniform(0.2, 0.8))
        return llm.call_likert_item(shared_ctx, profile, item_prompt)


def run_scoring_waves(units: list) -> dict:
    """Execute all item calls for the given units in fixed waves.

    units: [{"volunteer_id", "shared_ctx", "profile"}, ...] in matched-list
    order.  Returns {volunteer_id: {"raw_selections": [int×4]} |
    {"failed": True}}.  Futures are keyed (volunteer_id, item_idx) so
    aggregation is order-independent; an empty unit list spins up nothing
    (skip-on-empty is structural).
    """
    results: dict = {}
    for wave in partition_waves(units):
        futures = {}
        with ThreadPoolExecutor(max_workers=_SCORER_MAX_WORKERS) as pool:
            for unit in wave:
                for item_idx, item in enumerate(LIKERT_ITEMS):
                    futures[(unit["volunteer_id"], item_idx)] = pool.submit(
                        _score_item_with_retry,
                        unit["shared_ctx"], unit["profile"], item,
                    )
        for unit in wave:
            vid = unit["volunteer_id"]
            selections = []
            failed = False
            for item_idx in range(len(LIKERT_ITEMS)):
                try:
                    selections.append(futures[(vid, item_idx)].result())
                except Exception:
                    failed = True
            results[vid] = (
                {"failed": True} if failed
                else {"raw_selections": selections}
            )
    return results


def map_score_to_tier(total_score: Optional[int]) -> str:
    """Deterministic threshold mapping (S4).

    Attainable four-item sums are the nine even values in [−4, 12]:
    ≥ PERFECT_MIN → Perfect; ≥ GOOD_MIN → Good; ≤ 0 → Technical.  A None
    score (failure fallback) is Technical by policy.
    """
    if total_score is None:
        return "Technical Match"
    if total_score >= PERFECT_MIN:
        return "Perfect Match"
    if total_score >= GOOD_MIN:
        return "Good Match"
    return "Technical Match"


def compute_cap_reasons(state: dict, need_set: dict, vol_row) -> list:
    """Deterministic tier-cap inputs (S4).  Exactly two, per D8:
    (a) violation of the classifier's STATED soft preferences,
    (b) fix-6 schedule violation from the need-set description.
    Both run through the word-boundary detector; unconfirmed suggested
    skills are context only and never cap (a volunteer missing only
    suggestions can still reach Perfect — G5)."""
    reasons = []
    stated = str(state.get("soft_preferences", "") or "")
    if stated and matching.summarize_soft_preference_violations(
            {"description": stated}, vol_row):
        reasons.append("stated_soft_preference")
    if matching.summarize_soft_preference_violations(need_set, vol_row):
        reasons.append("schedule_preference")
    return reasons


def postprocess_recommendations(recs: list, caps_by_vid: dict,
                                names_by_vid: dict) -> list:
    """Assembly + caps + sort — the deterministic tail of the pipeline.

    - Threshold mapping for scored volunteers.  Almost Match is assigned
      upstream and never touched here; the failure fallback keeps its
      deterministic Technical Match.
    - Caps AFTER thresholds: a soft-preference violation caps the tier at
      Good Match.  Caps can only demote — a Technical volunteer with a
      violation stays Technical and records no applied cap.
    - Sort: tier rank, then preferred name (stable).
    """
    for rec in recs:
        if rec["tier"] == "Almost Match":
            continue
        rec["caps_applied"] = []
        if rec.get("raw_selections") is None:
            continue                       # failure fallback stays Technical
        rec["tier"] = map_score_to_tier(rec.get("total_score"))
        cap_reasons = caps_by_vid.get(rec["volunteer_id"], [])
        if cap_reasons and TIER_RANK[rec["tier"]] < TIER_RANK["Good Match"]:
            rec["tier"] = "Good Match"
            rec["caps_applied"] = list(cap_reasons)

    recs.sort(key=lambda r: (
        TIER_RANK.get(r["tier"], len(TIER_RANK)),
        str(names_by_vid.get(r["volunteer_id"], r["volunteer_id"])),
    ))
    return recs


def build_gap_notes(matched_volunteers: list, counterfactuals: dict) -> str:
    """Deterministic gap report (S5) — no LLM prose.

    One line per under-filled need set ("need N, found M") plus that need
    set's top counterfactual blockers with counts.  Empty string when
    every need set is covered.
    """
    lines = []
    for match_group in matched_volunteers:
        need = match_group.get("count_needed", 1)
        found = len(match_group.get("matched_volunteer_ids", []))
        if found >= need:
            continue
        desc = match_group.get("need_set_description", "")
        line = f"Need set '{desc}': need {need}, found {found}."
        prefix = f"NS{match_group.get('need_set_index')}: "
        blockers = sorted(
            (
                (key[len(prefix):], len(blocked))
                for key, blocked in counterfactuals.items()
                if key.startswith(prefix)
            ),
            key=lambda kv: (-kv[1], kv[0]),
        )
        if blockers:
            top = "; ".join(f"{name} blocks {n}" for name, n in blockers[:3])
            line += f" Top blockers: {top}."
        lines.append(line)
    return "\n".join(lines)


