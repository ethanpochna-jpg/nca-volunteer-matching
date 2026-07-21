"""On-demand reasoning: tier prompts, bundle builder, dissent detection.

Phase 4 verbatim move out of app.py (SECTION 7C).  Zero behavior edits;
cross-module references are import-qualified only.
"""

from core import llm, scoring

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
        scoring.build_volunteer_profile(vol_row),
        "",
        "=== ASSIGNED TIER ===",
        str(rec.get("tier", "")),
        "",
        "=== SCORE EVIDENCE (four Likert items, raw 1–5) ===",
    ]
    selections = rec.get("raw_selections")
    if selections:
        boxes = rec.get("boxes") or []
        for item, sel, box in zip(scoring.LIKERT_ITEMS, selections, boxes):
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
    text = llm.call_reasoning(bundle, REASONING_TIER_PROMPTS[tier])
    event = {
        "text": text,
        "dissent": detect_dissent(text),
        "tier": tier,
        "model": llm.REASONING_MODEL,
    }
    cache[cache_key] = event
    return event


