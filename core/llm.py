"""Anthropic client, the three call helpers, and the classifier prompt.

Phase 4 verbatim move out of app.py (SECTION 6 and SECTION 6B).  Zero
behavior edits; cross-module references are import-qualified only.
"""

import json
import os
from typing import Optional

import anthropic
import streamlit as st

from core import policy, schemas

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

Skills: {json.dumps(policy.VALID_SKILLS)}
Days: {json.dumps(policy.VALID_DAYS)}
Time blocks: {json.dumps(policy.VALID_TIME_BLOCKS)}
Languages: {json.dumps(policy.VALID_LANGUAGES)}
Areas: {json.dumps(policy.VALID_AREAS)}

Any value not in these lists will fail matching silently."""


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


def call_classifier(prompt_ctx: str) -> schemas.ClassifierOutput:
    """Opus 4.8 need-set extraction with grammar-constrained JSON output."""
    response = get_anthropic_client().messages.create(
        model=CLASSIFIER_MODEL,
        max_tokens=8192,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "medium",
            "format": {
                "type": "json_schema",
                "schema": _strict_schema(schemas.ClassifierOutput),
            },
        },
        system=CLASSIFIER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt_ctx}],
    )
    return schemas.ClassifierOutput.model_validate(json.loads(_first_text_block(response)))


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


