"""Pydantic schemas, LangGraph state shape, and state-serialization safety.

Phase 4 verbatim move out of app.py (SECTION 4, SECTION 5, and the safe
JSON helpers).  Zero behavior edits.
"""

import json
from datetime import date, datetime
from typing import TypedDict, Optional

import pandas as pd

# Pydantic enforces schema compliance on LLM outputs so that a malformed
# classifier response crashes loudly rather than silently producing wrong
# matches downstream.  This is the single most important guardrail in the
# system: if the LLM hallucinates a field name or type, Pydantic rejects it.
from pydantic import BaseModel, Field


# Safe JSON helpers for request-record serialization.
def json_safe_default(obj):
    try:
        import numpy as np
    except Exception:
        np = None

    # NumPy scalars
    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            val = float(obj)
            return None if pd.isna(val) else val
        if isinstance(obj, np.bool_):
            return bool(obj)

    # Pandas scalar/time types
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if obj is pd.NaT:
        return None

    # Standard library date/time
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # Pydantic models / generic scalar wrappers
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except Exception:
            pass

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def json_dumps_safe(value):
    return json.dumps(value, default=json_safe_default, ensure_ascii=False)


def sanitize_for_state(value):
    """Recursively convert pandas/NumPy scalars into msgpack-safe Python types."""
    try:
        return json.loads(json_dumps_safe(value))
    except Exception:
        if isinstance(value, dict):
            return {str(k): sanitize_for_state(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [sanitize_for_state(v) for v in value]
        return value



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PYDANTIC SCHEMAS FOR STRUCTURED LLM OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════
# These schemas are passed to model.with_structured_output() which leverages
# OpenAI function-calling to force the response into this exact shape.
# If the LLM produces invalid structure, it raises rather than silently
# propagating garbage downstream.

class FlexibleRequirement(BaseModel):
    """Represents a requirement with AND/OR logic.

    AND: all values must be satisfied simultaneously.
    OR:  at least one branch must be fully satisfied.
         Each branch is a list of values that must ALL hold together.

    Example — "Monday and either Saturday or both Sunday and Tuesday":
        AND: ["Mon"]
        OR:  [["Sat"], ["Sun", "Tue"]]

    Example — "English and Spanish, or English, Arabic, and Urdu":
        AND: []
        OR:  [["English", "Spanish"], ["English", "Arabic", "Urdu"]]

    Example — "must speak Spanish" (simple hard requirement):
        AND: ["Spanish"]
        OR:  []

    Example — no constraint at all:
        AND: []
        OR:  []
    """
    AND: list[str] = Field(default_factory=list, description="Values ALL required")
    OR: list[list[str]] = Field(
        default_factory=list,
        description="Disjunctive branches; at least one branch must be fully met."
    )


class NeedSet(BaseModel):
    """One distinct volunteer profile extracted from the user's request.

    If a user says "3 volunteers, at least one Spanish-speaking":
      → NeedSet(count=1, languages=FlexReq(AND=["Spanish"]), ...)
      → NeedSet(count=2, ...) (no language constraint)

    The classifier sorts need sets most-constrained first so that greedy
    pool-depletion in the matcher claims the scarcest matches first.
    """
    count: int = Field(
        description="How many volunteers needed with THIS specific profile"
    )
    description: str = Field(
        description="Natural-language summary of what this need set requires"
    )
    # Skills the LLM thinks are relevant — user confirms which are hard reqs
    applicable_skills: list[str] = Field(
        default_factory=list,
        description="Skills from the roster vocabulary that seem relevant. "
                    "User will confirm which are absolute requirements."
    )
    # Day availability with AND/OR logic
    availability_days: FlexibleRequirement = Field(
        default_factory=FlexibleRequirement
    )
    # Time-of-day blocks (simple list)
    availability_time_blocks: list[str] = Field(
        default_factory=list,
        description="Time blocks needed. Valid: Morning, Midday, Afternoon, Evening"
    )
    # Language requirements with AND/OR logic
    languages: FlexibleRequirement = Field(
        default_factory=FlexibleRequirement
    )
    # Minimum hours the task requires per session/week
    min_hours: Optional[float] = Field(
        default=None,
        description="Minimum hours per session/week the volunteer would commit"
    )
    # Preferred neighborhood (from roster vocabulary, or None)
    location_area: Optional[str] = Field(
        default=None,
        description="Preferred neighborhood. None = no preference."
    )
    # Whether a car is needed (for delivery/transport tasks)
    transportation_needed: Optional[str] = Field(
        default=None,
        description="'Car' if driving/delivery required, else null"
    )


class ClassifierOutput(BaseModel):
    """Top-level output from the needs-classifier LLM call.

    Contains the decomposed need sets plus a separate soft_preferences
    field that captures anything the user indicated as a preference
    rather than a hard requirement.  This separation (from app_6) is a
    KEY design choice: soft signals are forwarded to the recommender
    (not the matcher), preventing the "preferably Spanish → zero matches" bug.
    """
    need_sets: list[NeedSet] = Field(
        description="List of distinct volunteer profiles, most constrained first"
    )
    reasoning: str = Field(
        description="Brief explanation of how the request was decomposed"
    )
    # From app_6: explicit soft preferences extracted as plain text
    soft_preferences: str = Field(
        default="",
        description="Any preferences, nice-to-haves, or soft signals from the request "
                    "that should NOT be used for filtering but SHOULD inform the "
                    "recommender's tiering. Examples: 'preferably speaks Spanish', "
                    "'ideally someone experienced', 'would be nice if they have a car'."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LANGGRAPH STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════
# The state flows through the graph.  Each node reads the fields it needs
# and writes its outputs.  TypedDict documents the shape at definition time.

class GraphState(TypedDict):
    # ── User inputs (from the Streamlit form) ──
    user_prompt: str                       # Natural-language request text
    form_certs: list                       # Certifications selected in dropdown
    form_languages: list                   # Languages selected in dropdown
    has_specific_date: bool                # Whether a target date was set
    target_date: Optional[str]             # ISO date string, or None
    notification_date: str                 # ISO date for when volunteers are notified
    is_recurring: bool                     # One-time vs recurring
    recurring_end_date: Optional[str]      # If recurring, when does it end

    # ── Classifier outputs ──
    need_sets: list                        # List of NeedSet dicts
    extracted_skills: list                 # Skills the LLM suggested as relevant
    classifier_reasoning: str              # Explanation of the decomposition
    soft_preferences: str                  # Explicit soft preferences from classifier

    # ── Human-in-the-loop ──
    confirmed_skills: list                 # Skills the user confirmed as hard reqs
    unchecked_skills: list                 # Skills extracted but NOT confirmed (soft)

    # ── Matcher outputs ──
    matched_volunteers: list               # Per-need-set match results
    margins: dict                          # Per-volunteer capacity margins
    counterfactuals: dict                  # Per-requirement blocking analysis
    almost_matched: list                   # Volunteers blocked by exactly 1 req
    volunteer_histories: dict              # Assignment history per volunteer

    # ── Scoring outputs ──
    recommendations: list                  # Tiered, scored, capped, sorted
    gap_notes: Optional[str]               # Deterministic gap report (S5)

    # ── Request record ──
    request_record: dict                   # The full record written to CSV
