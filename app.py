"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Northbridge Community Alliance — Volunteer Matching & Coordination Assist. ║
║  GBA 479 Final Project Prototype  (v3 — Definitive)                        ║
║                                                                            ║
║  Architecture:  LangGraph state-graph with human-in-the-loop               ║
║  Models:        Anthropic — Opus 4.8 / Haiku 4.5 / Sonnet 4.6 (fixed)      ║
║  Interface:     Streamlit multi-step form                                  ║
║                                                                            ║
║  Graph flow:                                                               ║
║    [User Input] → classify_needs ──┐                                       ║
║                                    ├─ [Skills Confirmation by User] ──┐    ║
║                                    │                                  │    ║
║                     match_volunteers ◄────────────────────────────────┘    ║
║                           │                                                ║
║                     score_volunteers  (Likert waves)                       ║
║                           │                                                ║
║                     write_request_record                                   ║
║                           │                                                ║
║                        [Display]                                           ║
║                                                                            ║
║  Synthesizes the best patterns from three prior versions:                  ║
║    - v1 (app_4): Schema validation, per-need-set skill scoping,           ║
║          classifier output sanitization, NaN defaults on load,             ║
║          FlexibleRequirement normalization                                 ║
║    - v1 (app_5): Value aliasing/canonicalization, multi-delimiter          ║
║          parsing, recommender prompt discipline, Technical Match           ║
║          backfill, soft-preference violation detection, text truncation    ║
║    - v2 (app_6): Explicit soft_preferences in classifier schema,          ║
║          signal-word prompt engineering, dtype alignment, tier             ║
║          enforcement post-processing, rich display with NaN guards         ║
║                                                                            ║
║  New in v3:                                                                ║
║    1. Combined soft-preference strategy: explicit classifier field +       ║
║       unchecked-skills derivation + rule-based violation detection.        ║
║    2. Prompt compression: recommender receives only soft-fit-relevant      ║
║       fields; capacity/history/area are displayed deterministically.       ║
║    3. Per-need-set skill scoping (from v1-app4) merged with global         ║
║       vocabulary sanitization (from v1-app4) and canonicalization          ║
║       (from v1-app5).                                                      ║
║    4. Full postprocess pipeline: tier enforcement + backfill +             ║
║       soft-preference demotion + hallucinated-ID filtering.               ║
║    5. Defensive data loading: schema validation + NaN defaults +           ║
║       dtype alignment + volunteer_id stripping.                            ║
║                                                                            ║
║  Run:  streamlit run app.py                                                ║
║  Deps: pip install streamlit langgraph anthropic pandas openpyxl           ║
║        python-dotenv pydantic                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st          # UI framework for the multi-step form interface
import pandas as pd             # Tabular data handling for roster and assignments
import json                     # Serialization for LLM outputs and request records
import os                       # File-existence checks for data files
import random                   # Jitter for the scorer's single per-item retry
import re                       # Multi-delimiter parsing for roster fields
import sqlite3                  # Request-record persistence (stdlib — no ORM)
import time                     # Backoff sleep for the scorer's single retry
import uuid                     # Unique IDs for graph threads and request records
from concurrent.futures import ThreadPoolExecutor  # 16-call scoring waves
from datetime import date, datetime, timedelta  # Date arithmetic for notice periods
from typing import TypedDict, Optional, Literal # Type hints for graph state

# Pydantic enforces schema compliance on LLM outputs so that a malformed
# classifier response crashes loudly rather than silently producing wrong
# matches downstream.  This is the single most important guardrail in the
# system: if the LLM hallucinates a field name or type, Pydantic rejects it.
from pydantic import BaseModel, Field

# LangGraph manages the stateful orchestration graph.  StateGraph passes
# typed state between nodes; InMemorySaver checkpoints state so we can
# interrupt (for human-in-the-loop skills confirmation) and resume.
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# Native Anthropic SDK — every LLM stage (classifier / scorer / reasoning)
# goes through the helpers in SECTION 6B.  No LangChain LLM wrappers:
# structured output binds via native output_config.format (json_schema),
# which is the sanctioned path on a thinking-enabled call (PLAN §1b) —
# never forced tool choice.
import anthropic

# Load .env so ANTHROPIC_API_KEY is available without hardcoding it.
from dotenv import load_dotenv
load_dotenv()

# Phase 4 (in progress): symbols moved verbatim into core/ are imported
# back into this namespace so the remaining single-file code is untouched;
# the final UI-only commit strips these re-exports.
from core.schemas import (
    json_safe_default, json_dumps_safe, sanitize_for_state,
    FlexibleRequirement, NeedSet, ClassifierOutput, GraphState,
)
from core.policy import (
    VALID_SKILLS, VALID_CERTS_CLEARABLE, VALID_LANGUAGES, VALID_DAYS,
    VALID_TIME_BLOCKS, VALID_AREAS, VALUE_ALIASES, MANDATORY_CERT_RULES,
    YOUTH_FACING_SKILLS, FOOD_HANDLING_SKILLS, DRIVING_SKILLS,
    infer_mandatory_certs,
)
from core.matching import (
    ROSTER_PATH, ASSIGNMENTS_PATH, load_roster, load_assignments,
    canonicalize_value, truncate_text, summarize_soft_preference_violations,
    compute_volunteer_history, run_matching,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
# Every enum list below was extracted directly from the roster CSV.  The
# classifier and matcher both reference these so that vocabulary is aligned
# end-to-end: if a value isn't in these lists, it can't appear in the roster
# and would silently fail matching.

# requests.db is generated (gitignored), WAL-mode SQLite, seeded on first
# run by data/seed_requests.py.  Roster/assignments paths live in
# core.matching beside their loaders.
REQUESTS_DB_PATH = "requests.db"

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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

CLASSIFIER_SYSTEM_PROMPT = f"""You are the Needs Classifier for Northbridge Community Alliance's Volunteer Matching system.

YOUR JOB: Extract ONLY hard, non-negotiable volunteer requirements from a program manager's natural language request.

═══════════════════════════════════════════════════════════════
CRITICAL RULE — HARD vs SOFT DISTINCTION
═══════════════════════════════════════════════════════════════

This is the MOST IMPORTANT rule in your entire system prompt:

HARD requirements go into the need_sets schema fields.
  → "Must speak Spanish" → languages.AND: ["Spanish"]
  → "Need someone available Monday" → availability_days.AND: ["Mon"]

SOFT preferences go into the soft_preferences text field and NOWHERE ELSE.
  → "Preferably speaks Spanish" → soft_preferences: "Preferably speaks Spanish"
  → "Ideally available on Saturdays" → soft_preferences: "Ideally available on Saturdays"
  → "Would be nice if experienced" → soft_preferences: "Would be nice if experienced"

Signal words that mean SOFT (do NOT put in schema fields):
  preferably, ideally, would be nice, if possible, bonus if, nice to have,
  hoping for, it would help if, a plus, preferred, we'd love, would appreciate

Signal words that mean HARD (DO put in schema fields):
  must, need, required, has to, mandatory, necessary, essential, non-negotiable

When in doubt — leave it OUT of the hard schema and mention it in soft_preferences.
Over-constraining the hard schema means valid volunteers get filtered out and the
program manager sees zero matches when there were actually good candidates.

═══════════════════════════════════════════════════════════════
NEED SET DECOMPOSITION
═══════════════════════════════════════════════════════════════

- Produce one NeedSet per DISTINCT volunteer profile.
- "3 volunteers" with identical requirements → one NeedSet with count=3.
- "3 volunteers, at least one Spanish-speaking" → NeedSet(count=1, languages.AND=["Spanish"]) + NeedSet(count=2).
- Sort most-constrained need sets first (this matters for pool allocation).
- Merge duplicate profiles by summing their counts.

═══════════════════════════════════════════════════════════════
OR LOGIC — FlexibleRequirement format
═══════════════════════════════════════════════════════════════

Some fields (availability_days, languages) support AND/OR logic:
  AND: values that must ALL be present
  OR:  branches where at least ONE branch must be fully satisfied

Examples:
  "Monday and either Saturday or Sunday"
    → availability_days: {{AND: ["Mon"], OR: [["Sat"], ["Sun"]]}}
  "Tuesday and Thursday"
    → availability_days: {{AND: ["Tue", "Thu"], OR: []}}
  "Must speak English and Spanish, or English and Arabic"
    → languages: {{AND: [], OR: [["English", "Spanish"], ["English", "Arabic"]]}}
  "Must speak Spanish"
    → languages: {{AND: ["Spanish"], OR: []}}
  No day mentioned → availability_days: {{AND: [], OR: []}}
  "Preferably Monday" → availability_days: {{AND: [], OR: []}}
    (because "preferably" = soft, goes in soft_preferences instead)

═══════════════════════════════════════════════════════════════
APPLICABLE SKILLS
═══════════════════════════════════════════════════════════════

Suggest skills from the roster that seem relevant to the task described.
Be inclusive — suggest anything plausibly related. The user will narrow it
down in a confirmation step.  These are NOT yet hard requirements; they
become hard requirements only after the user explicitly confirms them.

═══════════════════════════════════════════════════════════════
VALID VOCABULARY (only use values from these lists)
═══════════════════════════════════════════════════════════════

Skills: {json.dumps(VALID_SKILLS)}
Days: {json.dumps(VALID_DAYS)}
Time blocks: {json.dumps(VALID_TIME_BLOCKS)}
Languages: {json.dumps(VALID_LANGUAGES)}
Areas: {json.dumps(VALID_AREAS)}

Any value not in these lists will fail matching silently."""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6B — ANTHROPIC CLIENT & CALL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
# All model access flows through these three helpers (PLAN §1a — fixed model
# matrix, no UI selector).  Design rules baked in:
#   - Classifier (Opus 4.8): explicit adaptive thinking + medium effort; NO
#     temperature (the parameter is rejected outright on Opus 4.8);
#     structured output via native output_config.format — never forced tool
#     choice on a thinking-enabled call.
#   - Scorer items (Haiku 4.5): temperature 0.2, tiny {selection: 1–5} schema.
#   - Reasoning (Sonnet 4.6): temperature 0.2, plain text, ~200 tokens.
# The client is a lazy singleton so importing this module never requires a
# key, and max_retries=0 so the scorer's single jittered retry (S3) is the
# only retry layer in the stack — deterministic under test.

CLASSIFIER_MODEL = "claude-opus-4-8"
SCORER_MODEL = "claude-haiku-4-5"
REASONING_MODEL = "claude-sonnet-4-6"

_ANTHROPIC_CLIENT: Optional[anthropic.Anthropic] = None


def _resolve_api_key() -> Optional[str]:
    """st.secrets first (Streamlit Cloud), environment second (local .env).

    st.secrets RAISES when no secrets file exists, so the try/except is
    load-bearing for local runs.
    """
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


def get_anthropic_client() -> anthropic.Anthropic:
    """Lazy client singleton.

    Tests monkeypatch THIS function with a mocked-transport client, so no
    code below it ever needs patching.
    """
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        _ANTHROPIC_CLIENT = anthropic.Anthropic(
            api_key=_resolve_api_key(),
            max_retries=0,
        )
    return _ANTHROPIC_CLIENT


def _strict_schema(model_cls) -> dict:
    """model_json_schema() with additionalProperties: false on every object
    node, as the structured-outputs grammar requires of all objects."""
    schema = json.loads(json.dumps(model_cls.model_json_schema()))

    def _walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node.setdefault("additionalProperties", False)
            for child in node.values():
                _walk(child)
        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(schema)
    return schema


def _first_text_block(response) -> str:
    """Adaptive thinking prepends thinking blocks; return the first text one."""
    for block in response.content:
        if block.type == "text":
            return block.text
    raise ValueError("Model response contained no text block")


def call_classifier(prompt_ctx: str) -> ClassifierOutput:
    """Opus 4.8 need-set extraction with grammar-constrained JSON output."""
    response = get_anthropic_client().messages.create(
        model=CLASSIFIER_MODEL,
        max_tokens=8192,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "medium",
            "format": {
                "type": "json_schema",
                "schema": _strict_schema(ClassifierOutput),
            },
        },
        system=CLASSIFIER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt_ctx}],
    )
    return ClassifierOutput.model_validate(json.loads(_first_text_block(response)))


_LIKERT_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {"selection": {"type": "integer", "enum": [1, 2, 3, 4, 5]}},
    "required": ["selection"],
    "additionalProperties": False,
}


def call_likert_item(shared_ctx: str, profile: str, item_text: str) -> int:
    """One Haiku 4.5 Likert judgment → raw selection 1–5.

    The model returns the raw selection only — box collapse, score mapping,
    and tier assignment are code-side (I4); the model never sees the word
    "tier" at scoring time.
    """
    response = get_anthropic_client().messages.create(
        model=SCORER_MODEL,
        max_tokens=256,
        temperature=0.2,
        output_config={
            "format": {"type": "json_schema", "schema": _LIKERT_SELECTION_SCHEMA},
        },
        system=shared_ctx,
        messages=[{"role": "user", "content": f"{profile}\n\n{item_text}"}],
    )
    return int(json.loads(_first_text_block(response))["selection"])


def call_reasoning(bundle: str, system_prompt: str) -> str:
    """One Sonnet 4.6 plain-text reasoning call.

    Transport only — the tier-conditional prompts and dissent detection
    layer on top in S6 (per-card button, outside the graph).
    """
    response = get_anthropic_client().messages.create(
        model=REASONING_MODEL,
        max_tokens=200,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": bundle}],
    )
    return _first_text_block(response).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7B — LIKERT SCORING PIPELINE (S3)
# ═══════════════════════════════════════════════════════════════════════════════
# Four Haiku 4.5 judgments per matched volunteer, executed in fixed waves.
# The LLM boundary is narrow by design: the model sees the request context,
# one volunteer profile, and one item's statement + anchors, and returns a
# raw 1–5 selection.  Everything else — box collapse, score mapping, tier
# thresholds, caps, ordering — is deterministic code (I1/I4).

# Concurrency is FIXED (no tuning knobs): 4 items × 4 volunteers in flight
# = 16 calls per wave; wave order is matched-list order.
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
  Skills: {truncate_text(vol_row['skills'])}
  Preferred Roles: {truncate_text(vol_row.get('preferred_roles', ''))}
  Certifications: {truncate_text(vol_row['certifications'])}
  Availability: {vol_row['availability_days']} / {vol_row['availability_time_blocks']}
  Availability Notes: {truncate_text(vol_row.get('availability_notes', ''))}
  Transportation: {vol_row.get('transportation', 'N/A')}
  Languages: {vol_row['languages']}
  Notes: {truncate_text(vol_row.get('notes', ''))}"""


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
        return call_likert_item(shared_ctx, profile, item_prompt)
    except Exception:
        time.sleep(random.uniform(0.2, 0.8))
        return call_likert_item(shared_ctx, profile, item_prompt)


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
    if stated and summarize_soft_preference_violations(
            {"description": stated}, vol_row):
        reasons.append("stated_soft_preference")
    if summarize_soft_preference_violations(need_set, vol_row):
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


def score_volunteers_node(state: GraphState) -> dict:
    """LLM node: four Likert items per matched volunteer, in waves.

    Tier assignment is deterministic: threshold mapping over the summed
    item scores, then caps, then ordering — all in
    postprocess_recommendations.  Raw selections, boxes, and totals ride
    along so the record never discards the distribution.  Almost-matched
    volunteers never enter the scorer (structural, not policy).
    """
    roster = load_roster()

    units = []
    caps_by_vid = {}
    names_by_vid = {}
    seen_vids = set()
    for match_group in state["matched_volunteers"]:
        need_set = state["need_sets"][match_group["need_set_index"]]
        shared_ctx = build_scorer_shared_context(state, need_set)
        for vid in match_group["matched_volunteer_ids"]:
            if vid in seen_vids:
                continue
            seen_vids.add(vid)
            vol_rows = roster[roster["volunteer_id"] == vid]
            if vol_rows.empty:
                continue
            vol = vol_rows.iloc[0]
            caps_by_vid[vid] = compute_cap_reasons(state, need_set, vol)
            names_by_vid[vid] = vol["preferred_name"]
            units.append({
                "volunteer_id": vid,
                "shared_ctx": shared_ctx,
                "profile": build_volunteer_profile(vol),
            })

    scores = run_scoring_waves(units)

    recs = []
    for unit in units:
        vid = unit["volunteer_id"]
        score = scores[vid]
        if score.get("failed"):
            recs.append({
                "volunteer_id": vid,
                "tier": "Technical Match",
                "reasoning": SCORING_UNAVAILABLE_NOTE,
                "raw_selections": None,
                "boxes": None,
                "total_score": None,
            })
            continue
        selections = score["raw_selections"]
        recs.append({
            "volunteer_id": vid,
            "tier": "Technical Match",     # replaced by the mapping below
            "reasoning": "",
            "raw_selections": selections,
            "boxes": [collapse_box(s) for s in selections],
            "total_score": sum(SCORE_MAP[s] for s in selections),
        })

    for am in state["almost_matched"]:
        vid = am["volunteer_id"]
        if vid in seen_vids:
            continue
        seen_vids.add(vid)
        names_by_vid[vid] = am.get("preferred_name", vid)
        recs.append({
            "volunteer_id": vid,
            "tier": "Almost Match",
            "reasoning": (
                f"Blocked by exactly one hard requirement: "
                f"{am['blocking_requirement']}."
            ),
        })

    recs = postprocess_recommendations(recs, caps_by_vid, names_by_vid)

    return sanitize_for_state({
        "recommendations": recs,
        "gap_notes": build_gap_notes(
            state["matched_volunteers"], state.get("counterfactuals", {})
        ) or None,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7C — ON-DEMAND REASONING (S6)
# ═══════════════════════════════════════════════════════════════════════════════
# Per-card "Get reasoning" — one Sonnet 4.6 call, OUTSIDE the graph (D9).
# The model explains the code-assigned tier from its score evidence; it can
# disagree ("On second thought…"), which is LOGGED as dissent but never
# mutates the tier (I5).  Almost Match cards carry templated blocker text
# and get no button (D-H).

# Tier-conditional prompts — Perfect is Ethan's verbatim text; Good and
# Technical are the approved variants in his pattern (PLAN §S6).
REASONING_TIER_PROMPTS = {
    "Perfect Match": (
        "Review why this respondent is a perfect fit for the request, "
        "rather than just a good fit. If you believe they are not a "
        "perfect fit, preface your reasoning with, \"On second "
        "thought...\". Explain their fit by highlighting how their "
        "profile aligns with the request in 1-2 sentences."
    ),
    "Good Match": (
        "Review why this respondent is a good fit for the request, rather "
        "than a perfect fit or a merely technical one. If you believe "
        "this tier is wrong in either direction, preface your reasoning "
        "with, \"On second thought...\". Explain their fit by "
        "highlighting how their profile aligns with the request, and "
        "what keeps them short of a perfect fit, in 1-2 sentences."
    ),
    "Technical Match": (
        "Review why this respondent technically qualifies for the request "
        "but may not be a natural fit. If you believe they are a stronger "
        "fit than a technical match, preface your reasoning with, \"On "
        "second thought...\". Explain what qualifies them and where the "
        "misalignment lies in 1-2 sentences."
    ),
}

_DISSENT_PREFIX = "on second thought"


def detect_dissent(text: str) -> bool:
    """D-G: reasoning BEGINNING "On second thought" (case-insensitive,
    tolerant of leading straight/curly quote marks and whatever
    punctuation follows the phrase) flags dissent.  A mid-text mention is
    NOT dissent.  The flag is logged; the tier never changes (I5)."""
    normalized = str(text or "").lstrip()
    normalized = normalized.lstrip("\"'‘’“”‛`").lstrip()
    return normalized.casefold().startswith(_DISSENT_PREFIX)


def build_reasoning_bundle(user_prompt: str, need_set_desc: str,
                           soft_preferences: str, vol_row, rec: dict) -> str:
    """Everything the reasoning model may cite: request summary, need-set
    description, stated soft preferences, the volunteer's compressed
    profile, the assigned tier, and the item results — the model explains
    the tier from its evidence."""
    lines = [
        "=== REQUEST SUMMARY ===",
        str(user_prompt or ""),
        "",
        "=== NEED SET ===",
        str(need_set_desc or ""),
        "",
        "=== STATED SOFT PREFERENCES ===",
        soft_preferences if soft_preferences else "None stated.",
        "",
        build_volunteer_profile(vol_row),
        "",
        "=== ASSIGNED TIER ===",
        str(rec.get("tier", "")),
        "",
        "=== SCORE EVIDENCE (four Likert items, raw 1–5) ===",
    ]
    selections = rec.get("raw_selections")
    if selections:
        boxes = rec.get("boxes") or []
        for item, sel, box in zip(LIKERT_ITEMS, selections, boxes):
            lines.append(f"{item['key']}: {sel} ({box})")
        lines.append(f"Total score: {rec.get('total_score')}")
    else:
        lines.append("Automated scoring was unavailable for this volunteer.")
    return "\n".join(lines)


def fetch_reasoning(bundle: str, tier: str, cache: dict, cache_key) -> dict:
    """Fetch (or reuse) the reasoning event for one card.

    The cache is injected — the UI passes an st.session_state-backed dict
    keyed (thread_id, volunteer_id); tests pass a plain dict — so rerun
    deduplication is unit-testable.  Returns the event dict that S7 logs
    to reasoning_events.
    """
    if cache_key in cache:
        return cache[cache_key]
    text = call_reasoning(bundle, REASONING_TIER_PROMPTS[tier])
    event = {
        "text": text,
        "dissent": detect_dissent(text),
        "tier": tier,
        "model": REASONING_MODEL,
    }
    cache[cache_key] = event
    return event


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7D — REQUEST-RECORD PERSISTENCE (S7)
# ═══════════════════════════════════════════════════════════════════════════════
# stdlib sqlite3, WAL mode, short-lived connections, writes in transactions.
# The requests table is BORN at schema_version 2 (D3 — no CSV, no v1, no
# migration; nothing existed before this).  Schema changes from here on
# require bumping schema_version and a migration note in the commit body.
# reasoning_events is APPEND-ONLY: INSERT is the only statement ever
# issued against it — dissent rate is a queryable QA metric for the
# future Insights Agent.

SCHEMA_VERSION = 2

_REQUESTS_COLUMNS = [
    "request_id", "schema_version", "timestamp", "user_prompt",
    "soft_preferences", "unchecked_skills", "request_source",
    "need_sets_json", "confirmed_skills_json", "extracted_skills_json",
    "form_certs_json", "form_languages_json", "has_specific_date",
    "target_date", "notification_date", "is_recurring",
    "matched_volunteers_json", "margins_json", "counterfactuals_json",
    "almost_matched_json", "recommendations_json", "gap_notes",
    "resulting_assignment_ids",
]


def db_connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Short-lived WAL connection; every writer opens its own."""
    conn = sqlite3.connect(db_path or REQUESTS_DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_request_db(db_path: Optional[str] = None) -> None:
    """Create both tables if absent.  Idempotent."""
    with db_connect(db_path) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL DEFAULT {SCHEMA_VERSION},
                timestamp TEXT NOT NULL,
                user_prompt TEXT NOT NULL,
                soft_preferences TEXT,
                unchecked_skills TEXT,
                request_source TEXT,
                need_sets_json TEXT,
                confirmed_skills_json TEXT,
                extracted_skills_json TEXT,
                form_certs_json TEXT,
                form_languages_json TEXT,
                has_specific_date INTEGER,
                target_date TEXT,
                notification_date TEXT,
                is_recurring INTEGER,
                matched_volunteers_json TEXT,
                margins_json TEXT,
                counterfactuals_json TEXT,
                almost_matched_json TEXT,
                recommendations_json TEXT,
                gap_notes TEXT,
                resulting_assignment_ids TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                volunteer_id TEXT NOT NULL,
                tier TEXT NOT NULL,
                model TEXT NOT NULL,
                text TEXT NOT NULL,
                dissent INTEGER NOT NULL CHECK (dissent IN (0, 1)),
                created_at TEXT NOT NULL
            )
        """)


def insert_request_record(record: dict, db_path: Optional[str] = None) -> None:
    """One request row, one transaction."""
    placeholders = ", ".join(f":{col}" for col in _REQUESTS_COLUMNS)
    with db_connect(db_path) as conn:
        conn.execute(
            f"INSERT INTO requests ({', '.join(_REQUESTS_COLUMNS)}) "
            f"VALUES ({placeholders})",
            record,
        )


def log_reasoning_event(request_id: str, volunteer_id: str, event: dict,
                        db_path: Optional[str] = None) -> None:
    """Append one reasoning event — one row per button fetch.

    INSERT only; this table is never UPDATEd and never DELETEd from.
    """
    with db_connect(db_path) as conn:
        conn.execute(
            "INSERT INTO reasoning_events "
            "(request_id, volunteer_id, tier, model, text, dissent, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                request_id,
                volunteer_id,
                event["tier"],
                event["model"],
                event["text"],
                1 if event.get("dissent") else 0,
                datetime.now().isoformat(),
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — LANGGRAPH NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def classify_needs_node(state: GraphState) -> dict:
    """LLM node: extracts structured volunteer requirements from natural language.

    Calls Opus 4.8 through call_classifier (SECTION 6B) — native
    structured outputs guarantee the ClassifierOutput schema.
    KEY FEATURES:
    - From app_6: soft_preferences field in schema + signal-word prompt
    - From app_4: post-extraction vocabulary sanitization
    - From app_5: skill validation against VALID_SKILLS
    """
    context_parts = [f"Request: {state['user_prompt']}"]
    if state["form_certs"]:
        context_parts.append(
            f"Required certifications (already specified via form): {state['form_certs']}"
        )
    if state["form_languages"]:
        context_parts.append(
            f"Required languages (already specified via form): {state['form_languages']}"
        )
    if state["has_specific_date"]:
        context_parts.append(f"Target date: {state['target_date']}")
    if state.get("is_recurring"):
        context_parts.append(
            f"This is a recurring need (ends: {state.get('recurring_end_date', 'TBD')})"
        )

    user_msg = "\n".join(context_parts)

    result = call_classifier(user_msg)

    # ── Collect and validate extracted skills ────────────────────────────
    # Fix 10: canonicalize before membership testing — a classifier emitting
    # "tutoring - math" must not be silently dropped from the confirmation
    # UI (never confirmable → never enforced → never cert-triggering).
    all_skills = []
    seen = set()
    for ns in result.need_sets:
        for sk in ns.applicable_skills:
            csk = canonicalize_value(sk, "skills")
            if csk and csk not in seen and csk in VALID_SKILLS:
                all_skills.append(csk)
                seen.add(csk)

    # ── Sanitize ALL vocabulary fields in each need set (from app_4) ─────
    # The LLM can produce near-miss strings like "spanish" instead of
    # "Spanish" or "monday" instead of "Mon" — these silently fail matching
    # because Python string comparison is case-sensitive.  We validate and
    # canonicalize every field.
    valid_langs_set = set(VALID_LANGUAGES)
    valid_days_set = set(VALID_DAYS)
    valid_time_set = set(VALID_TIME_BLOCKS)
    valid_areas_set = set(VALID_AREAS)

    sanitized_need_sets = []
    for ns in result.need_sets:
        ns_dict = ns.model_dump()

        # Validate languages in FlexibleRequirement
        lang_req = ns_dict.get("languages", {"AND": [], "OR": []})
        lang_req["AND"] = [
            canonicalize_value(v, "languages")
            for v in lang_req.get("AND", [])
            if canonicalize_value(v, "languages") in valid_langs_set
        ]
        lang_req["OR"] = [
            [canonicalize_value(v, "languages") for v in branch
             if canonicalize_value(v, "languages") in valid_langs_set]
            for branch in lang_req.get("OR", [])
        ]
        lang_req["OR"] = [b for b in lang_req["OR"] if b]
        ns_dict["languages"] = lang_req

        # Validate days in FlexibleRequirement
        days_req = ns_dict.get("availability_days", {"AND": [], "OR": []})
        days_req["AND"] = [
            canonicalize_value(v, "days")
            for v in days_req.get("AND", [])
            if canonicalize_value(v, "days") in valid_days_set
        ]
        days_req["OR"] = [
            [canonicalize_value(v, "days") for v in branch
             if canonicalize_value(v, "days") in valid_days_set]
            for branch in days_req.get("OR", [])
        ]
        days_req["OR"] = [b for b in days_req["OR"] if b]
        ns_dict["availability_days"] = days_req

        # Validate time blocks
        ns_dict["availability_time_blocks"] = [
            canonicalize_value(v, "time_blocks")
            for v in ns_dict.get("availability_time_blocks", [])
            if canonicalize_value(v, "time_blocks") in valid_time_set
        ]

        # Validate location area
        if ns_dict.get("location_area") and ns_dict["location_area"] not in valid_areas_set:
            ns_dict["location_area"] = None

        # Validate applicable_skills (fix 10: canonicalize like other domains)
        canonical_skills = []
        for s in ns_dict.get("applicable_skills", []):
            cs = canonicalize_value(s, "skills")
            if cs in VALID_SKILLS and cs not in canonical_skills:
                canonical_skills.append(cs)
        ns_dict["applicable_skills"] = canonical_skills

        sanitized_need_sets.append(ns_dict)

    return {
        "need_sets": sanitized_need_sets,
        "extracted_skills": all_skills,
        "classifier_reasoning": result.reasoning,
        "soft_preferences": result.soft_preferences,
    }


def match_volunteers_node(state: GraphState) -> dict:
    """Deterministic node: filters volunteers against hard requirements.

    No LLM here — pure Python constraint satisfaction.  Runs matching for
    each need set independently using a greedy approach.

    KEY FIX from app_4: per-need-set skill scoping.  Each need set only
    enforces skills that the classifier assigned to THIS need set AND the
    user confirmed globally.  This prevents "one driver + one intake
    volunteer" from requiring both Driver and Intake on every slot.
    """
    roster = load_roster()
    assignments = load_assignments()

    # ── Pass 1 (fix 3): every need set's matched pool vs the FULL roster ──
    # Greedy claiming in roster order strands feasible assignments (executed
    # proof: the only Spanish+Car volunteer was claimed by the Spanish slot,
    # leaving the driver slot unfilled although a two-slot assignment
    # existed).  These full pools feed the scarcity ranking below; the
    # authoritative matching still runs against the depleted roster.
    full_pools: list[set] = []
    for need_set in state["need_sets"]:
        ns_applicable = set(need_set.get("applicable_skills", []))
        ns_confirmed = [s for s in state["confirmed_skills"] if s in ns_applicable]
        pool_result = run_matching(
            need_set=need_set,
            confirmed_skills=ns_confirmed,
            form_certs=state["form_certs"],
            form_languages=state["form_languages"],
            has_specific_date=state["has_specific_date"],
            target_date_str=state["target_date"],
            notification_date_str=state["notification_date"],
            roster_df=roster,
            assignments_df=assignments,
        )
        full_pools.append(set(pool_result["matched"]))

    all_matched = []
    all_margins = {}
    all_counterfactuals = {}
    all_almost = []
    all_histories = {}
    claimed_vids = set()

    for ns_idx, need_set in enumerate(state["need_sets"]):
        available_roster = roster[~roster["volunteer_id"].isin(claimed_vids)]

        # ── Per-need-set skill scoping (from app_4) ────────────────────
        # Only enforce skills that BOTH: (a) the classifier assigned to
        # this specific need set, AND (b) the user confirmed globally.
        ns_applicable = set(need_set.get("applicable_skills", []))
        ns_confirmed_skills = [
            s for s in state["confirmed_skills"] if s in ns_applicable
        ]

        match_result = run_matching(
            need_set=need_set,
            confirmed_skills=ns_confirmed_skills,
            form_certs=state["form_certs"],
            form_languages=state["form_languages"],
            has_specific_date=state["has_specific_date"],
            target_date_str=state["target_date"],
            notification_date_str=state["notification_date"],
            roster_df=available_roster,
            assignments_df=assignments,
        )

        ns_matched = match_result["matched"]
        all_matched.append({
            "need_set_index": ns_idx,
            "need_set_description": need_set.get("description", ""),
            "count_needed": need_set.get("count", 1),
            "matched_volunteer_ids": ns_matched,
            # Fix 5: margins stored per need-set group (rides inside
            # matched_volunteers_json — no new record column).
            "margins": match_result["margins"],
        })

        # Fix 5: the flat dict is display-only and FIRST-wins.  Need sets
        # are ordered most-constrained first, so the first group's margins
        # are the meaningful ones; update() let a later, laxer need set
        # overwrite them (executed proof: Spanish displayed as an "extra"
        # language against the need set that REQUIRED Spanish).
        for vid, margin in match_result["margins"].items():
            all_margins.setdefault(vid, margin)

        for req_name, blocked in match_result["counterfactuals"].items():
            key = f"NS{ns_idx}: {req_name}"
            all_counterfactuals[key] = blocked

        all_almost.extend(match_result["almost_matched"])

        # ── Pass 2 (fix 3): claim by scarcity, not roster order ────────
        # Volunteers useful to the FEWEST other need sets are claimed
        # first, preserving multi-pool volunteers for the slots that can
        # only be filled by them.  Tie-break is pool order, so single-
        # need-set behavior is byte-identical to the old greedy claim.
        def _other_pool_count(vid, _idx=ns_idx):
            return sum(
                1 for j, pool in enumerate(full_pools)
                if j != _idx and vid in pool
            )

        claim_order = sorted(
            ns_matched,
            key=lambda v: (_other_pool_count(v), ns_matched.index(v)),
        )
        for vid in claim_order[:need_set.get("count", 1)]:
            claimed_vids.add(vid)

    # Compute assignment history for all relevant volunteers
    relevant_vids = set()
    for m in all_matched:
        relevant_vids.update(m["matched_volunteer_ids"])
    for am in all_almost:
        relevant_vids.add(am["volunteer_id"])

    for vid in relevant_vids:
        all_histories[vid] = compute_volunteer_history(vid, assignments)

    return sanitize_for_state({
        "matched_volunteers": all_matched,
        "margins": all_margins,
        "counterfactuals": all_counterfactuals,
        "almost_matched": all_almost,
        "volunteer_histories": all_histories,
    })


def write_request_record_node(state: GraphState) -> dict:
    """Terminal node: persists the full request record to SQLite (S7).

    The record captures the full pipeline (I6) — extraction,
    confirmations, matches, margins, counterfactuals, raw Likert
    selections, boxes, totals, caps, tiers, and the deterministic gap
    notes.  It is the future Insights Agent's input.
    """
    record = {
        "request_id": str(uuid.uuid4()),
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now().isoformat(),
        "user_prompt": state["user_prompt"],
        "soft_preferences": state.get("soft_preferences", ""),
        "unchecked_skills": json_dumps_safe(state.get("unchecked_skills", [])),
        "request_source": "user_input",
        "need_sets_json": json_dumps_safe(state["need_sets"]),
        "confirmed_skills_json": json_dumps_safe(state["confirmed_skills"]),
        "extracted_skills_json": json_dumps_safe(state["extracted_skills"]),
        "form_certs_json": json_dumps_safe(state["form_certs"]),
        "form_languages_json": json_dumps_safe(state["form_languages"]),
        "has_specific_date": bool(state["has_specific_date"]),
        "target_date": state.get("target_date", ""),
        "notification_date": state["notification_date"],
        "is_recurring": bool(state.get("is_recurring", False)),
        "matched_volunteers_json": json_dumps_safe(state["matched_volunteers"]),
        "margins_json": json_dumps_safe(state["margins"]),
        "counterfactuals_json": json_dumps_safe(state["counterfactuals"]),
        "almost_matched_json": json_dumps_safe(state["almost_matched"]),
        "recommendations_json": json_dumps_safe(state["recommendations"]),
        "gap_notes": state.get("gap_notes") or "",
        "resulting_assignment_ids": "[]",
    }

    init_request_db()
    insert_request_record(record)

    return sanitize_for_state({"request_record": record})


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """Construct the LangGraph state-graph with human-in-the-loop interrupt.

    The graph pauses after classify_needs so the user can confirm which
    extracted skills should be treated as hard requirements.  When resumed,
    it runs match_volunteers → score_volunteers → write_request_record.
    """
    builder = StateGraph(GraphState)

    builder.add_node("classify_needs", classify_needs_node)
    builder.add_node("match_volunteers", match_volunteers_node)
    builder.add_node("score_volunteers", score_volunteers_node)
    builder.add_node("write_request_record", write_request_record_node)

    builder.add_edge(START, "classify_needs")
    builder.add_edge("classify_needs", "match_volunteers")
    builder.add_edge("match_volunteers", "score_volunteers")
    builder.add_edge("score_volunteers", "write_request_record")
    builder.add_edge("write_request_record", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["match_volunteers"],
    )

    return graph


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — STREAMLIT USER INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Entry point: page config, sidebar, session state, stage dispatch."""

    st.set_page_config(
        page_title="NCA Volunteer Matching Assistant",
        page_icon="🤝",
        layout="wide",
    )

    st.title("🤝 Northbridge Volunteer Matching Assistant")
    st.caption(
        "AI-powered volunteer coordination for program managers. "
        "Describe what you need — the system identifies, filters, and recommends volunteers."
    )

    # ── Sidebar: read-only configuration (D-J: no model selector) ──────
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("**Models in use**")
        st.caption(f"Classifier: `{CLASSIFIER_MODEL}`")
        st.caption(f"Scorer: `{SCORER_MODEL}`")
        st.caption(f"Reasoning: `{REASONING_MODEL}`")

        st.divider()
        st.markdown("**Data Files**")
        st.caption(f"Roster: `{ROSTER_PATH}`")
        st.caption(f"Assignments: `{ASSIGNMENTS_PATH}`")
        st.caption(f"Requests DB: `{REQUESTS_DB_PATH}`")
        st.caption("Demo dataset — resets on redeploy.")

    # ── Session state initialization ───────────────────────────────────
    if "stage" not in st.session_state:
        st.session_state["stage"] = "input"
    if "graph" not in st.session_state:
        st.session_state["graph"] = build_graph()
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())

    # ── Verify data files exist ────────────────────────────────────────
    if not os.path.exists(ROSTER_PATH):
        st.error(f"Roster file not found: `{ROSTER_PATH}`. Place it in the app directory.")
        st.stop()
    if not os.path.exists(ASSIGNMENTS_PATH):
        st.error(
            f"Assignments file not found: `{ASSIGNMENTS_PATH}`. "
            f"Place it in the app directory."
        )
        st.stop()

    # ── Seed the demo request history on first run (S7) ────────────────
    # requests.db is generated and gitignored; a fresh deploy (or reboot
    # of the ephemeral container) rebuilds it from the seed script.
    if not os.path.exists(REQUESTS_DB_PATH):
        from data.seed_requests import seed_database
        init_request_db()
        with db_connect() as conn:
            seed_database(conn)

    # ── Stage dispatch ─────────────────────────────────────────────────
    if st.session_state["stage"] == "input":
        render_input_stage()
    elif st.session_state["stage"] == "skills_review":
        render_skills_review_stage()
    elif st.session_state["stage"] == "results":
        render_results_stage()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — INPUT
# ═══════════════════════════════════════════════════════════════════════════════

def render_input_stage():
    """Render the initial request input form."""

    st.subheader("📋 Describe Your Volunteer Need")

    user_prompt = st.text_area(
        "What do you need?",
        placeholder=(
            "Example: I need 2 volunteers for Saturday morning pantry — one "
            "should speak Spanish for intake, and both need to be able to do "
            "sorting and stocking."
        ),
        height=120,
        help="Describe the role, timing, skills, and any special requirements "
             "in plain language. Use 'must' for hard requirements and "
             "'preferably' for nice-to-haves.",
    )

    st.subheader("📌 Hard Requirements")
    st.caption(
        "These override and supplement the natural language extraction. "
        "Anything selected here is treated as a non-negotiable filter."
    )

    col1, col2 = st.columns(2)

    with col1:
        form_certs = st.multiselect(
            "Required Certifications",
            options=VALID_CERTS_CLEARABLE,
            help="Only volunteers with ALL selected certifications will match. "
                 "Policy-mandated certs (e.g., Background Check for tutoring) "
                 "are added automatically based on skills.",
        )

    with col2:
        form_languages = st.multiselect(
            "Required Languages",
            options=VALID_LANGUAGES,
            help="Only volunteers who speak ALL selected languages will match.",
        )

    st.subheader("📅 Scheduling")
    col_date, col_notify = st.columns(2)

    with col_date:
        has_specific_date = st.checkbox("I have a specific target date")
        target_date = None
        if has_specific_date:
            target_date = st.date_input(
                "Target Date",
                value=date.today() + timedelta(days=7),
                min_value=date.today(),
            )

    with col_notify:
        notification_date = st.date_input(
            "Notification Date (when volunteers will be contacted)",
            value=date.today(),
        )

    # ── Recurring toggle ───────────────────────────────────────────────
    is_recurring = st.checkbox("This is a recurring need")
    recurring_end_date = None
    if is_recurring:
        recurring_end_date = st.date_input(
            "Recurring until",
            value=date.today() + timedelta(days=90),
        )

    # ── Submit ─────────────────────────────────────────────────────────
    if st.button("🔍 Analyze Request", type="primary", use_container_width=True):
        if not user_prompt.strip():
            st.warning("Please describe your volunteer need before submitting.")
            return

        # Fix 4: date-pair guard.  A notification date after the target
        # date drives the notice window negative, which silently blocks
        # the entire roster on the (deliberately hard) Notice Period
        # check — surface it at submit instead.
        if has_specific_date and target_date and notification_date > target_date:
            st.error(
                "Notification date is after the target date — the notice "
                "window would be negative and no volunteer could match. "
                "Adjust one of the dates."
            )
            return
        if notification_date < date.today():
            st.warning(
                "Notification date is in the past — backdating inflates "
                "the notice window and can overstate volunteer availability."
            )

        initial_state = {
            "user_prompt": user_prompt.strip(),
            "form_certs": form_certs,
            "form_languages": form_languages,
            "has_specific_date": has_specific_date,
            "target_date": str(target_date) if target_date else None,
            "notification_date": str(notification_date),
            "is_recurring": is_recurring,
            "recurring_end_date": str(recurring_end_date) if recurring_end_date else None,
        }

        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        st.session_state["graph_config"] = config

        with st.spinner("🧠 Analyzing your request..."):
            try:
                st.session_state["graph"].invoke(initial_state, config)
            except Exception as e:
                st.error(f"Classification failed: {e}")
                return

        # Retrieve state after classifier completes (graph is paused)
        snapshot = st.session_state["graph"].get_state(config)
        st.session_state["classifier_state"] = snapshot.values
        st.session_state["stage"] = "skills_review"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — SKILLS REVIEW (Human-in-the-Loop)
# ═══════════════════════════════════════════════════════════════════════════════

def render_skills_review_stage():
    """Render the skills confirmation step.

    The user reviews the classifier's extracted skills and checks which
    ones should be treated as hard requirements.  Unchecked skills become
    soft preferences forwarded to the recommender.
    """

    state = st.session_state["classifier_state"]

    st.subheader("🔍 Review Extracted Requirements")

    # Show classifier reasoning
    with st.expander("🧠 Classifier Reasoning", expanded=True):
        st.write(state["classifier_reasoning"])

    # Show soft preferences if any were extracted
    if state.get("soft_preferences"):
        st.info(f"**Soft preferences detected:** {state['soft_preferences']}")

    # Fix 4: make the notice window visible before matching runs, so a
    # thin result set is explainable against volunteers' minimum notice.
    if state.get("has_specific_date") and state.get("target_date"):
        try:
            _target = date.fromisoformat(str(state["target_date"]))
            _notify = date.fromisoformat(str(state["notification_date"]))
            st.caption(
                f"🗓️ Notice window: {(_target - _notify).days} day(s) "
                f"between notification and target date."
            )
        except (ValueError, TypeError):
            pass

    # Show need sets
    with st.expander("📦 Need Sets", expanded=True):
        for i, ns in enumerate(state["need_sets"]):
            st.markdown(f"**Need Set {i + 1}:** {ns['description']} (count: {ns['count']})")
            details = []
            days = ns.get("availability_days", {})
            if days.get("AND") or days.get("OR"):
                details.append(f"Days: AND={days.get('AND', [])}, OR={days.get('OR', [])}")
            blocks = ns.get("availability_time_blocks", [])
            if blocks:
                details.append(f"Time blocks: {blocks}")
            langs = ns.get("languages", {})
            if langs.get("AND") or langs.get("OR"):
                details.append(f"Languages: AND={langs.get('AND', [])}, OR={langs.get('OR', [])}")
            if ns.get("min_hours"):
                details.append(f"Min hours: {ns['min_hours']}")
            if ns.get("location_area"):
                details.append(f"Location (informational only): {ns['location_area']}")
            if ns.get("transportation_needed"):
                details.append(f"Transportation: {ns['transportation_needed']}")
            for d in details:
                st.caption(f"  {d}")

    # ── Skills confirmation ────────────────────────────────────────────
    st.subheader("✅ Confirm Required Skills")
    st.caption(
        "Check skills that are **absolute requirements** — volunteers without "
        "them will be filtered out.  Unchecked skills will be treated as "
        "preferences that inform the recommender's ranking."
    )

    extracted = state.get("extracted_skills", [])
    if not extracted:
        st.info("No skills were extracted from your request.")
        confirmed = []
    else:
        confirmed = []
        cols = st.columns(min(len(extracted), 3))
        for i, skill in enumerate(extracted):
            with cols[i % len(cols)]:
                if st.checkbox(skill, value=False, key=f"skill_{skill}"):
                    confirmed.append(skill)

    # Compute unchecked skills for soft-preference forwarding
    unchecked = [s for s in extracted if s not in confirmed]

    # Show what mandatory certs will be auto-added.  Fix 1 mirror: computed
    # from extracted ∪ confirmed — the same work-type basis the matcher
    # enforces — so the user sees the certs even with nothing checked.
    auto_certs = infer_mandatory_certs(sorted(set(extracted) | set(confirmed)))
    if auto_certs:
        st.caption(
            f"🔒 Auto-added certifications based on the type of work "
            f"identified: {', '.join(auto_certs)}"
        )

    # ── Action buttons ─────────────────────────────────────────────────
    col_back, col_confirm = st.columns(2)

    with col_back:
        if st.button("← Back to Input", use_container_width=True):
            st.session_state["stage"] = "input"
            st.session_state["thread_id"] = str(uuid.uuid4())
            st.session_state["graph"] = build_graph()
            st.rerun()

    with col_confirm:
        if st.button("✅ Confirm & Match", type="primary", use_container_width=True):
            config = st.session_state["graph_config"]

            # Update graph state with confirmed skills AND unchecked (soft)
            st.session_state["graph"].update_state(
                config,
                {
                    "confirmed_skills": confirmed,
                    "unchecked_skills": unchecked,
                },
            )

            with st.spinner("🔄 Matching volunteers and generating recommendations..."):
                try:
                    final_state = st.session_state["graph"].invoke(None, config)
                except Exception as e:
                    st.error(f"Matching/recommendation failed: {e}")
                    return

            st.session_state["final_state"] = final_state
            st.session_state["stage"] = "results"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_results_stage():
    """Render the final tiered recommendations and analysis.

    Rich display from app_6: pronouns handling, availability notes,
    NaN guards, conditional margin labeling based on whether a specific
    date was provided.
    """

    state = st.session_state["final_state"]
    roster = load_roster()

    st.subheader("📊 Volunteer Recommendations")

    # First need set that matched each volunteer — context for reasoning
    # bundles (consistent with first-wins margins).
    ns_desc_by_vid = {}
    for mg in state.get("matched_volunteers", []):
        for _vid in mg.get("matched_volunteer_ids", []):
            ns_desc_by_vid.setdefault(_vid, mg.get("need_set_description", ""))

    # ── Gap notes (deterministic, S5) ──────────────────────────────────
    if state.get("gap_notes"):
        st.warning(f"**Gap report:** {state['gap_notes']}")

    # ── Tiered recommendations ─────────────────────────────────────────
    recommendations = state.get("recommendations", [])

    tier_order = ["Perfect Match", "Good Match", "Technical Match", "Almost Match"]
    tier_icons = {
        "Perfect Match": "🌟",
        "Good Match": "👍",
        "Technical Match": "⚙️",
        "Almost Match": "⚠️",
    }
    tier_colors = {
        "Perfect Match": "#e8f5e9",
        "Good Match": "#e3f2fd",
        "Technical Match": "#fff3e0",
        "Almost Match": "#fce4ec",
    }

    for tier in tier_order:
        tier_recs = [r for r in recommendations if r["tier"] == tier]
        if not tier_recs:
            continue

        st.markdown(f"### {tier_icons.get(tier, '')} {tier} ({len(tier_recs)})")

        for rec in tier_recs:
            vid = rec["volunteer_id"]
            vol_rows = roster[roster["volunteer_id"] == vid]
            if vol_rows.empty:
                continue
            vol = vol_rows.iloc[0]
            margin = state.get("margins", {}).get(vid, {})
            history = state.get("volunteer_histories", {}).get(vid, {})

            with st.container():
                st.markdown(
                    f"<div style='background-color:{tier_colors.get(tier, '#f5f5f5')}; "
                    f"padding:12px; border-radius:8px; margin-bottom:8px;'>",
                    unsafe_allow_html=True,
                )

                # Header: name, ID, pronouns (with NaN guard)
                pronouns = vol.get("pronouns", "")
                pronouns_str = f" · {pronouns}" if pronouns and str(pronouns) not in ("nan", "", "NaN") else ""
                st.markdown(f"**{vol['preferred_name']}** ({vid}){pronouns_str}")

                # ── Reasoning (S6) ─────────────────────────────────
                if tier == "Almost Match":
                    # D-H: templated blocker text inline, no button.
                    st.caption(rec["reasoning"])
                else:
                    cache = st.session_state.setdefault("reasoning_cache", {})
                    cache_key = (st.session_state.get("thread_id"), vid)
                    cached = cache.get(cache_key)
                    if cached:
                        # I5/D-G: text shown verbatim; dissent is logged,
                        # never applied — the tier above stays as scored.
                        st.caption(cached["text"])
                    elif rec.get("reasoning"):
                        st.caption(rec["reasoning"])  # scoring-unavailable note

                    if st.button("💬 Get reasoning", key=f"reason_{vid}"):
                        bundle = build_reasoning_bundle(
                            state.get("user_prompt", ""),
                            ns_desc_by_vid.get(vid, ""),
                            state.get("soft_preferences", ""),
                            vol,
                            rec,
                        )
                        fresh = cache_key not in cache
                        with st.spinner("Fetching reasoning..."):
                            event = fetch_reasoning(bundle, tier, cache, cache_key)
                        # S7: one reasoning_events row per button FETCH
                        # (reruns replay from cache and log nothing).
                        request_id = state.get("request_record", {}).get("request_id")
                        if fresh and request_id:
                            log_reasoning_event(request_id, vid, event)
                        st.rerun()

                # ── Volunteer details in 3 columns ─────────────────
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("**Availability**")
                    st.text(f"Days: {vol['availability_days']}")
                    st.text(f"Time: {vol['availability_time_blocks']}")
                    avail_notes = vol.get("availability_notes", "")
                    if avail_notes and str(avail_notes) not in ("nan", "", "NA"):
                        st.text(f"Notes: {avail_notes}")

                with c2:
                    st.markdown("**Profile**")
                    st.text(f"Area: {vol['home_area']}")
                    transport = vol.get("transportation", "N/A")
                    st.text(f"Transport: {transport if str(transport) != 'nan' else 'N/A'}")
                    st.text(f"Languages: {vol['languages']}")
                    pref_roles = vol.get("preferred_roles", "")
                    if pref_roles and str(pref_roles) not in ("nan", "", "NA"):
                        st.text(f"Preferred roles: {pref_roles}")
                    notes = vol.get("notes", "")
                    if notes and str(notes) not in ("nan", "", "NA"):
                        st.text(f"Notes: {notes}")

                with c3:
                    # ── Capacity Margins ────────────────────────────
                    if tier != "Almost Match" and margin:
                        st.markdown("**Capacity Margins**")

                        hrs_rem = margin.get("hours_remaining_after_request")
                        max_hrs = margin.get("max_hours_per_week", "?")
                        committed = margin.get("hours_committed_this_week", 0)

                        if margin.get("has_specific_date"):
                            if hrs_rem is not None:
                                st.text(
                                    f"Hours remaining: {hrs_rem} "
                                    f"(of {max_hrs}/wk, {committed} committed)"
                                )
                            else:
                                st.text(f"Max hours/wk: {max_hrs}")
                            notice_slack = margin.get("notice_slack_days")
                            if notice_slack is not None:
                                st.text(f"Notice slack: {notice_slack} days")
                        else:
                            st.text(f"Max hours/wk: {max_hrs}")
                            vol_notice = margin.get("vol_min_notice_days", "?")
                            st.text(f"Min notice required: {vol_notice} days")

                        extra_skills = margin.get("extra_skills", [])
                        if extra_skills:
                            st.text(f"Extra skills: {', '.join(extra_skills)}")

                    # ── Assignment History ──────────────────────────
                    st.markdown("**History**")
                    total_asgn = history.get("total_assignments", 0)
                    no_shows = history.get("no_shows", 0)
                    no_show_rate = history.get("no_show_rate", 0)
                    last_asgn = history.get("last_assigned", "No prior assignments")
                    st.text(f"Assignments: {total_asgn}")
                    st.text(f"No-shows: {no_shows} ({no_show_rate:.0%})")
                    st.text(f"Last assigned: {last_asgn}")

                st.markdown("</div>", unsafe_allow_html=True)

    # Handle zero recommendations
    if not recommendations:
        st.warning(
            "No volunteer recommendations were generated.  This may mean "
            "no volunteers matched the specified requirements.  Consider "
            "relaxing some constraints and trying again."
        )

    # ── Counterfactual analysis (expandable) ───────────────────────────
    counterfactuals = state.get("counterfactuals", {})
    if counterfactuals:
        with st.expander(
            "📈 Counterfactual Analysis — Per-Requirement Blocking",
            expanded=False,
        ):
            st.caption(
                "For each hard requirement, shows volunteers who would have "
                "matched if that single requirement were relaxed."
            )
            for req_name, blocked_list in counterfactuals.items():
                st.markdown(f"**{req_name}**")
                for bv in blocked_list:
                    st.text(
                        f"  {bv['preferred_name']} ({bv['volunteer_id']}) "
                        f"— blocked by: {bv['blocking_column']}"
                    )

    # ── Need set match summary (expandable) ────────────────────────────
    with st.expander("📋 Match Summary by Need Set", expanded=False):
        for mg in state.get("matched_volunteers", []):
            st.markdown(
                f"**{mg['need_set_description']}** "
                f"(need {mg['count_needed']}, "
                f"found {len(mg['matched_volunteer_ids'])})"
            )
            if mg["matched_volunteer_ids"]:
                st.text(f"  Matched: {', '.join(mg['matched_volunteer_ids'])}")
            else:
                st.text("  No matches found.")

    # ── Request record (expandable) ────────────────────────────────────
    with st.expander(
        "💾 Request Record (written to requests data)", expanded=False
    ):
        record = state.get("request_record", {})
        if record:
            st.json(record)
        else:
            st.caption("No record generated.")

    # ── New request button ─────────────────────────────────────────────
    st.divider()
    if st.button("🔄 New Request", type="primary", use_container_width=True):
        st.session_state["stage"] = "input"
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.session_state["graph"] = build_graph()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
