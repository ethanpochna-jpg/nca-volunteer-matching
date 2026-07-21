"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Northbridge Community Alliance — Volunteer Matching & Coordination Assist. ║
║  GBA 479 Final Project Prototype  (v3 — Definitive)                        ║
║                                                                            ║
║  Architecture:  LangGraph state-graph with human-in-the-loop               ║
║  Model:         OpenAI GPT (configurable in sidebar)                       ║
║  Interface:     Streamlit multi-step form                                  ║
║                                                                            ║
║  Graph flow:                                                               ║
║    [User Input] → classify_needs ──┐                                       ║
║                                    ├─ [Skills Confirmation by User] ──┐    ║
║                                    │                                  │    ║
║                     match_volunteers ◄────────────────────────────────┘    ║
║                           │                                                ║
║                     recommend_volunteers                                   ║
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
║  Deps: pip install streamlit langgraph langchain-openai pandas openpyxl    ║
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
import re                       # Multi-delimiter parsing for roster fields
import uuid                     # Unique IDs for graph threads and request records
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

# LangChain's OpenAI wrapper provides .with_structured_output(), which uses
# OpenAI function-calling under the hood to guarantee schema-valid JSON.
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load .env so OPENAI_API_KEY is available without hardcoding it.
from dotenv import load_dotenv
load_dotenv()


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
# SECTION 2 — CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
# Every enum list below was extracted directly from the roster CSV.  The
# classifier and matcher both reference these so that vocabulary is aligned
# end-to-end: if a value isn't in these lists, it can't appear in the roster
# and would silently fail matching.

# File paths — data files live under data/ at the repo root; the app and
# the test suite both run from the repo root, so relative paths suffice.
ROSTER_PATH = "data/northbridge_volunteer_roster.csv"
ASSIGNMENTS_PATH = "data/northbridge_volunteer_assignments.xlsx"
REQUESTS_DATA_PATH = "northbridge_requests_data.csv"

# ── Roster vocabulary ─────────────────────────────────────────────────────

VALID_SKILLS = [
    "Adult Learning", "Analytics/Reporting", "Community Outreach",
    "Crafts/Activities", "Customer Service", "Data Entry", "Driver",
    "ESL Support", "Environmental Cleanup", "Event Support",
    "Forklift (experienced)", "Intake/Translation", "Inventory/Sorting",
    "Pantry Operations", "Photography/Media", "Program Support",
    "Scheduling/Coordination", "Team Lead", "Tool Safety",
    "Training/Onboarding", "Tutoring - Math", "Tutoring - Reading",
    "Tutoring - SAT Prep", "Tutoring - Science", "Volunteer Training",
    "Warehouse/Lifting", "Youth Mentoring",
]

# Only "cleared"/"approved"/"completed" certifications count as held.
# "Pending" variants mean the volunteer is NOT yet certified and must be
# treated as if the cert is absent for matching purposes.
VALID_CERTS_CLEARABLE = [
    "Background Check - Cleared",
    "Child Safety Training - Completed",
    "Food Safety - Basic",
    "Driver Authorization - Approved",
    "Tool Safety Briefing - Completed",
    "Trainer - Approved",
    "Waiver - Signed",
]

VALID_LANGUAGES = [
    "Arabic", "Bengali", "English", "French", "Japanese", "Korean",
    "Russian", "Spanish", "Swahili", "Tamil", "Twi", "Urdu", "Vietnamese",
]

VALID_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

VALID_TIME_BLOCKS = ["Morning", "Midday", "Afternoon", "Evening"]

VALID_AREAS = [
    "Northbridge - Downtown", "Northbridge - Eastview",
    "Northbridge - Maplewood", "Northbridge - Riverbend",
    "Northbridge - South Market", "Northbridge - Westside",
]

# ── Value aliases (from app_5) ───────────────────────────────────────────
# Maps natural-language variants and common misspellings to canonical roster
# vocabulary.  The canonicalize_value() function uses these to normalize
# LLM outputs and user inputs before matching.
VALUE_ALIASES = {
    "languages": {
        "spanish-speaking": "Spanish", "spanish speaking": "Spanish",
        "speak spanish": "Spanish",
        "arabic-speaking": "Arabic", "arabic speaking": "Arabic",
        "bengali-speaking": "Bengali", "bengali speaking": "Bengali",
        "english-speaking": "English", "english speaking": "English",
        "french-speaking": "French", "french speaking": "French",
        "japanese-speaking": "Japanese", "japanese speaking": "Japanese",
        "korean-speaking": "Korean", "korean speaking": "Korean",
        "russian-speaking": "Russian", "russian speaking": "Russian",
        "swahili-speaking": "Swahili", "swahili speaking": "Swahili",
        "tamil-speaking": "Tamil", "tamil speaking": "Tamil",
        "twi-speaking": "Twi", "twi speaking": "Twi",
        "urdu-speaking": "Urdu", "urdu speaking": "Urdu",
        "vietnamese-speaking": "Vietnamese", "vietnamese speaking": "Vietnamese",
    },
    "days": {
        "monday": "Mon", "mon": "Mon",
        "tuesday": "Tue", "tue": "Tue", "tues": "Tue",
        "wednesday": "Wed", "wed": "Wed",
        "thursday": "Thu", "thu": "Thu", "thurs": "Thu",
        "friday": "Fri", "fri": "Fri",
        "saturday": "Sat", "sat": "Sat",
        "sunday": "Sun", "sun": "Sun",
    },
    "time_blocks": {
        "am": "Morning", "morning": "Morning",
        "midday": "Midday", "mid-day": "Midday", "noon": "Midday",
        "lunch": "Midday",
        "afternoon": "Afternoon", "pm": "Afternoon",
        "evening": "Evening", "night": "Evening",
    },
    "transportation": {
        "car": "Car", "vehicle": "Car", "own car": "Car",
        "public transit": "Public Transit", "transit": "Public Transit",
        "bus": "Public Transit", "train": "Public Transit",
        "bike": "Bike", "bicycle": "Bike",
        "walk": "Walk", "walking": "Walk",
        "any": "Any",
    },
}

# ── Mandatory certification rules (Coordination Manual, Section 3) ────────
# These are ORGANIZATIONAL policy, not per-request.  The matcher enforces
# them automatically when the confirmed skills trigger a category.
MANDATORY_CERT_RULES = {
    "youth_facing": [                           # Any tutoring / youth role
        "Background Check - Cleared",
        "Child Safety Training - Completed",
    ],
    "food_handling": [                          # Pantry / food contact roles
        "Food Safety - Basic",
    ],
    "driving": [                                # Delivery / transport roles
        "Driver Authorization - Approved",
    ],
}

# Which skills trigger which mandatory-cert category.
YOUTH_FACING_SKILLS = {
    "Tutoring - Math", "Tutoring - Reading", "Tutoring - Science",
    "Tutoring - SAT Prep", "ESL Support", "Youth Mentoring",
    "Crafts/Activities",
}
FOOD_HANDLING_SKILLS = {
    "Pantry Operations", "Inventory/Sorting", "Intake/Translation",
}
DRIVING_SKILLS = {"Driver"}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
# Defensive loading: schema validation (from app_4), dtype alignment (from
# app_6), NaN defaults (from app_4), and volunteer_id stripping (from app_6).

# ── Required columns for schema validation ────────────────────────────────
REQUIRED_ROSTER_COLUMNS = [
    "volunteer_id", "preferred_name", "skills", "certifications", "languages",
    "availability_days", "availability_time_blocks", "max_hours_per_week",
    "min_notice_days", "status", "preferred_roles", "home_area", "transportation",
]
REQUIRED_ASSIGNMENT_COLUMNS = [
    "volunteer_id", "status", "start_date", "hours_required",
]


def validate_dataframe(df: pd.DataFrame, required_cols: list, file_label: str) -> list:
    """Check that a DataFrame contains all required columns.

    Returns list of missing column names (empty if all present).
    """
    return [c for c in required_cols if c not in df.columns]


def normalize_assignment_status(value) -> str:
    """Normalize assignment status variants to canonical internal values."""
    if value is None or pd.isna(value):
        return ""
    raw = str(value).strip().casefold().replace('-', '_').replace(' ', '_')
    status_map = {
        'confirmed': 'confirmed',
        'complete': 'completed',
        'completed': 'completed',
        'done': 'completed',
        'no_show': 'no_show',
        'noshow': 'no_show',
        'cancelled': 'cancelled',
        'canceled': 'cancelled',
    }
    return status_map.get(raw, raw)


@st.cache_data
def load_roster() -> pd.DataFrame:
    """Load the volunteer roster CSV with defensive typing and defaults.

    Key fixes consolidated from all three prior versions:
    1. Read as dtype=str to prevent int/float volunteer_id mismatches (app_6).
    2. Validate required columns and halt early if any are missing (app_4).
    3. Coerce numeric columns explicitly; fill NaN with permissive defaults
       so that volunteers with missing data aren't silently excluded (app_4).
    4. Strip whitespace from volunteer_id to prevent silent join failures (app_6).
    """
    df = pd.read_csv(ROSTER_PATH, dtype=str)

    # Schema validation — fail fast if the CSV is malformed
    missing = validate_dataframe(df, REQUIRED_ROSTER_COLUMNS, "Roster")
    if missing:
        st.error(f"Roster file is missing required columns: {missing}")
        st.stop()

    # Coerce numeric columns; errors='coerce' turns unparseable → NaN
    df["max_hours_per_week"] = pd.to_numeric(df["max_hours_per_week"], errors="coerce")
    df["min_notice_days"] = pd.to_numeric(df["min_notice_days"], errors="coerce")

    # Permissive NaN defaults: if we don't know their constraints, assume
    # no constraint.  This prevents volunteers with missing data from being
    # silently excluded by the matcher.
    df["min_notice_days"] = df["min_notice_days"].fillna(0)
    df["max_hours_per_week"] = df["max_hours_per_week"].fillna(40)

    # Strip whitespace from volunteer_id to prevent silent join failures
    df["volunteer_id"] = df["volunteer_id"].str.strip()

    return df


@st.cache_data
def load_assignments() -> pd.DataFrame:
    """Load the volunteer assignments dataset (XLSX or CSV).

    All columns as string first, then coerce hours_required to float.
    """
    if ASSIGNMENTS_PATH.endswith(".xlsx"):
        df = pd.read_excel(ASSIGNMENTS_PATH, sheet_name="Assignments", dtype=str)
    else:
        df = pd.read_csv(ASSIGNMENTS_PATH, dtype=str)

    # Schema validation
    missing = validate_dataframe(df, REQUIRED_ASSIGNMENT_COLUMNS, "Assignments")
    if missing:
        st.error(f"Assignments file is missing required columns: {missing}")
        st.stop()

    # Align dtypes and normalize common workbook inconsistencies
    df["volunteer_id"] = df["volunteer_id"].astype(str).str.strip()
    df["status"] = df["status"].apply(normalize_assignment_status)
    df["hours_required"] = pd.to_numeric(df["hours_required"], errors="coerce").fillna(0.0)
    df["start_date_parsed"] = pd.to_datetime(df["start_date"], errors="coerce")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3B — VALUE CANONICALIZATION & PARSING
# ═══════════════════════════════════════════════════════════════════════════════
# From app_5: absorbs formatting inconsistencies in both roster data and LLM
# outputs by mapping natural-language variants to canonical vocabulary.

def canonicalize_value(value, domain: Optional[str] = None) -> str:
    """Normalize a value to canonical roster vocabulary.

    1. Handles NaN, None, "NA", empty strings → returns "".
    2. If a domain is specified, checks VALUE_ALIASES for known mappings.
    3. Falls back to case-insensitive match against the valid list.
    4. Returns the original string if no match is found.
    """
    if value is None or pd.isna(value):
        return ""
    raw = str(value).strip()
    if not raw or raw in ("NA", "nan", "NaN"):
        return ""
    if domain is None:
        return raw

    # Check alias map first (handles "Spanish-speaking" → "Spanish" etc.)
    alias_map = VALUE_ALIASES.get(domain, {})
    raw_lower = raw.casefold()
    if raw_lower in alias_map:
        return alias_map[raw_lower]

    # Fall back to case-insensitive match against valid values
    valid_values = {
        "languages": VALID_LANGUAGES,
        "days": VALID_DAYS,
        "time_blocks": VALID_TIME_BLOCKS,
        "skills": VALID_SKILLS,
    }.get(domain, [])
    for valid in valid_values:
        if raw_lower == valid.casefold():
            return valid

    return raw


def parse_semicolon(value, domain: Optional[str] = None) -> set:
    """Split a multi-valued roster field into a normalized set.

    From app_5: tolerates semicolons, commas, pipes, and newlines as
    delimiters, making the parser robust to small formatting inconsistencies
    in the roster CSV or user exports.  Each token is canonicalized.

    Returns an empty set for NaN, "NA", empty strings, and None.
    """
    if value is None or pd.isna(value):
        return set()
    raw = str(value).strip()
    if not raw or raw in ("NA", "nan", "NaN"):
        return set()
    parts = [p.strip() for p in re.split(r"[;|\n,]+", raw) if p.strip()]
    normalized = {canonicalize_value(p, domain) for p in parts}
    normalized.discard("")
    return normalized


def truncate_text(value, max_len: int = 500) -> str:
    """Trim long note fields so one candidate cannot dominate the prompt.

    From app_5: prevents a single volunteer's extensive notes from consuming
    a disproportionate share of the recommender's context window.
    """
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


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


class VolunteerRecommendation(BaseModel):
    """A single volunteer's recommendation entry from the recommender."""
    volunteer_id: str
    # Literal type constraint (from app_4/5) prevents the LLM from inventing tiers
    tier: Literal["Perfect Match", "Good Match", "Technical Match", "Almost Match"] = Field(
        description="One of: Perfect Match, Good Match, Technical Match, Almost Match"
    )
    reasoning: str = Field(
        description="Concise explanation citing specific data points"
    )


class RecommenderOutput(BaseModel):
    """Top-level output from the recommendation LLM call."""
    recommendations: list[VolunteerRecommendation]
    configuration_notes: Optional[str] = Field(
        default=None,
        description="If multiple volunteers needed, notes on optimal grouping"
    )
    gap_notes: Optional[str] = Field(
        default=None,
        description="If insufficient matches, explain gaps and suggest next steps"
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

    # ── Recommender outputs ──
    recommendations: list                  # Tiered recommendations (validated)
    configuration_notes: Optional[str]     # Multi-volunteer grouping notes
    gap_notes: Optional[str]               # Gap report if insufficient matches

    # ── Request record ──
    request_record: dict                   # The full record written to CSV


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


# ── Recommender system prompt ─────────────────────────────────────────────
# From app_5's design: purely instructional, no dynamic data.  All request-
# specific and volunteer-specific data goes in the human message.  The prompt
# is explicit about what the LLM should NOT use for ranking, since those
# factors are handled deterministically and displayed separately.

RECOMMENDER_SYSTEM_PROMPT = """You are the Volunteer Recommendation Specialist for Northbridge Community Alliance.

ROLE:
You receive already-eligible matched volunteers plus almost-matched volunteers.
All matched volunteers have already passed the hard filters. Rank only on soft fit.

ONLY USE THESE FACTORS FOR RANKING MATCHED VOLUNTEERS:
- Role fit / preferred roles alignment with the task
- Relevant skills and certifications beyond the minimum
- Language fit (especially when soft preferences mention language)
- Transportation fit
- Availability days and time blocks, especially interpreting soft scheduling preferences
- Natural-language notes and availability notes
- Explicit soft preference flags provided in the input

DO NOT USE THESE AS RANKING FACTORS (they are handled deterministically):
- Capacity metrics (hours remaining, max hours)
- Assignment history (total assignments, no-show rate)
- Home area
These are displayed to the user separately and should not influence your tiering.

═══════════════════════════════════════════════════════════════
TIER DEFINITIONS — assign exactly one tier per volunteer
═══════════════════════════════════════════════════════════════

- Perfect Match: FROM MATCHED GROUP ONLY. Strongest soft fit. Preferred roles
  align with the task, availability is natural, soft preferences are satisfied,
  notes suggest enthusiasm for this type of work. This is who you'd call first.
  A volunteer CANNOT be Perfect Match if they violate any soft preference.

- Good Match: FROM MATCHED GROUP ONLY. Strong fit, but weaker than Perfect.
  May be missing one or more soft preferences but is otherwise solid. A volunteer
  who lacks soft-preference attributes cannot be Perfect Match — Good Match at best.

- Technical Match: FROM MATCHED GROUP ONLY. Passes hard requirements but the role
  doesn't align with their stated preferences, or schedule is tight, or this feels
  outside what they signed up for.

- Almost Match: FROM ALMOST-MATCHED GROUP ONLY. Blocked by exactly one hard
  requirement. Explain specifically what's blocking them and whether it's resolvable
  (e.g., "pending certification expected in 2 weeks").

CRITICAL: A volunteer from the ALMOST-MATCHED group must ALWAYS be tiered as
Almost Match regardless of how good their soft fit appears. No exceptions.

═══════════════════════════════════════════════════════════════
IMPORTANT RULES
═══════════════════════════════════════════════════════════════

- You do not need to include every matched volunteer. A deterministic post-processor
  will add any remaining matched volunteers as Technical Match.
- Keep reasoning concise and evidence-based — cite specific data points.
- Only use volunteer_ids that appear in the data provided. Do not invent IDs.
- Provide configuration_notes for multi-volunteer requests when useful.
- Provide gap_notes when the pool is thin or insufficient.

COMMUNICATION TONE (per NCA Brand Guidelines):
- Respectful, clear, evidence-informed, practical, non-assumptive.
- Emphasize appreciation and impact when discussing volunteers.
- Avoid over-promising or guaranteeing availability."""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — DETERMINISTIC MATCHING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
# This is the core of the system — pure Python, no LLM involvement.
# Every check corresponds to an organizational rule from the Coordination
# Manual or a constraint from the roster schema.

def evaluate_flexible_requirement(requirement: dict, volunteer_values: set) -> bool:
    """Check whether a volunteer satisfies a FlexibleRequirement.

    Evaluation rule:
      - ALL values in the AND list must be present in volunteer_values.
      - At least ONE branch in the OR list must be fully satisfied
        (every value in that branch present in volunteer_values).
      - If both AND and OR are empty, the requirement is vacuously satisfied.
    """
    # ── AND portion: every listed value must be in the volunteer's set ──
    and_vals = requirement.get("AND", [])
    if and_vals:
        real_and = [v for v in and_vals if v and v not in ("NA", "")]
        if real_and and not all(v in volunteer_values for v in real_and):
            return False

    # ── OR portion: at least one branch must be fully met ──
    or_branches = requirement.get("OR", [])
    if or_branches:
        branch_satisfied = False
        for branch in or_branches:
            branch_vals = branch if isinstance(branch, list) else [branch]
            if all(v in volunteer_values for v in branch_vals if v):
                branch_satisfied = True
                break
        if not branch_satisfied:
            return False

    return True


def normalize_flexible_requirement(requirement: dict, domain: Optional[str] = None) -> dict:
    """Canonicalize and deduplicate a FlexibleRequirement.

    From app_4: removes AND values from OR branches (since AND values are
    guaranteed, they're redundant in OR).  Collapses empty branches.
    From app_5: canonicalizes each value against the domain vocabulary.
    """
    # Canonicalize AND values
    and_set = set()
    and_list = []
    for v in requirement.get("AND", []) or []:
        cv = canonicalize_value(v, domain)
        if cv and cv not in and_set:
            and_set.add(cv)
            and_list.append(cv)

    # Canonicalize OR branches.  Two distinct empty-branch cases (fix 2):
    # a branch with no canonical values at all is garbage and is dropped
    # alone, but a branch whose values are all covered by AND is satisfied
    # whenever AND holds — the OR clause is then vacuous and EVERY branch
    # must clear.  Dropping only the subsumed branch would leave its
    # siblings mandatory and make the requirement strictly harder
    # ("Monday (or Monday and Saturday)" must not become Mon AND Sat).
    or_branches = []
    seen_branches = set()
    for branch in requirement.get("OR", []) or []:
        branch_vals = branch if isinstance(branch, list) else [branch]
        canonical = []
        branch_seen = set()
        for v in branch_vals:
            cv = canonicalize_value(v, domain)
            if not cv or cv in branch_seen:
                continue
            branch_seen.add(cv)
            canonical.append(cv)
        if not canonical:
            continue                # garbage-only branch → drop the branch
        remaining = [cv for cv in canonical if cv not in and_set]
        if not remaining:
            or_branches = []        # branch implied by AND → OR is vacuous
            break
        branch_key = tuple(remaining)
        if branch_key in seen_branches:
            continue
        seen_branches.add(branch_key)
        or_branches.append(remaining)

    return {"AND": and_list, "OR": or_branches}


def get_all_required_values(requirement: dict) -> set:
    """Extract every value mentioned anywhere in a FlexibleRequirement.

    Used for margin calculations — tells us the full universe of values
    the request "cares about" so we can compute what the volunteer has beyond.
    """
    vals = set(requirement.get("AND", []))
    for branch in requirement.get("OR", []):
        branch_list = branch if isinstance(branch, list) else [branch]
        vals.update(branch_list)
    vals.discard("NA")
    vals.discard("")
    return vals


def infer_mandatory_certs(confirmed_skills: list) -> list:
    """Determine which certifications are ORGANIZATIONALLY REQUIRED.

    These are non-negotiable policy rules from the Coordination Manual:
    - Youth-facing skills → Background Check + Child Safety Training
    - Food handling skills → Food Safety
    - Driving skills → Driver Authorization
    """
    mandatory = set()
    skill_set = set(confirmed_skills)

    if skill_set & YOUTH_FACING_SKILLS:
        mandatory.update(MANDATORY_CERT_RULES["youth_facing"])
    if skill_set & FOOD_HANDLING_SKILLS:
        mandatory.update(MANDATORY_CERT_RULES["food_handling"])
    if skill_set & DRIVING_SKILLS:
        mandatory.update(MANDATORY_CERT_RULES["driving"])

    return list(mandatory)


def get_committed_hours(
    vol_id: str,
    target_date_str: Optional[str],
    assignments_df: pd.DataFrame,
) -> float:
    """Calculate hours a volunteer is already committed to in the target week.

    Uses ISO week boundaries (Mon–Sun).  Returns 0.0 if no target date.
    """
    if not target_date_str:
        return 0.0

    try:
        target = date.fromisoformat(target_date_str)
    except (ValueError, TypeError):
        return 0.0

    iso_year, iso_week, _ = target.isocalendar()
    week_start = date.fromisocalendar(iso_year, iso_week, 1)
    week_end = date.fromisocalendar(iso_year, iso_week, 7)

    mask = (
        (assignments_df["volunteer_id"] == vol_id)
        & (assignments_df["status"].isin(["confirmed", "completed"]))
    )
    vol_assignments = assignments_df.loc[mask]

    total_hours = 0.0
    for _, row in vol_assignments.iterrows():
        try:
            parsed = row.get("start_date_parsed")
            if pd.notna(parsed):
                a_date = parsed.date()
            else:
                a_date = date.fromisoformat(str(row["start_date"])[:10])
            if week_start <= a_date <= week_end:
                total_hours += float(row["hours_required"])
        except (ValueError, TypeError, AttributeError):
            continue

    return total_hours


def compute_volunteer_history(vol_id: str, assignments_df: pd.DataFrame) -> dict:
    """Compute assignment history summary for a volunteer.

    From app_6's fix: handles dtype alignment by ensuring string comparison.
    From app_4: uses pd.to_datetime for correct date ordering.
    Returns sensible defaults when a volunteer has zero assignments.
    """
    vol_rows = assignments_df[assignments_df["volunteer_id"] == str(vol_id).strip()]

    total = len(vol_rows)
    completed = len(vol_rows[vol_rows["status"] == "completed"])
    no_shows = len(vol_rows[vol_rows["status"] == "no_show"])
    cancelled = len(vol_rows[vol_rows["status"] == "cancelled"])

    # Most recent assignment date (prefer parsed column if present)
    dates = vol_rows["start_date_parsed"] if "start_date_parsed" in vol_rows.columns else pd.to_datetime(vol_rows["start_date"], errors="coerce")
    if dates.notna().any():
        last_assigned = str(dates.max().date())
    else:
        last_assigned = "No prior assignments"

    return {
        "total_assignments": total,
        "completed": completed,
        "no_shows": no_shows,
        "cancelled": cancelled,
        "no_show_rate": round(no_shows / max(total, 1), 3),
        "last_assigned": last_assigned,
    }


def summarize_soft_preference_violations(need_set: dict, vol_row: pd.Series) -> list[str]:
    """Return rule-based soft-preference violation notes.

    From app_5: detects schedule-preference violations by keyword parsing
    on the need-set description.  This is a heuristic layer that doesn't
    depend on the LLM, providing defense-in-depth for the tiering logic.
    """
    desc = str(need_set.get("description", "") or "")
    desc_l = desc.lower()
    if not any(token in desc_l for token in ["prefer", "preferred", "ideally"]):
        return []

    violations = []
    vol_days = parse_semicolon(vol_row.get("availability_days", ""), domain="days")
    vol_blocks = parse_semicolon(
        vol_row.get("availability_time_blocks", ""), domain="time_blocks"
    )

    day_map = VALUE_ALIASES.get("days", {})
    for raw, norm in day_map.items():
        if f"prefer {raw}" in desc_l or f"ideally {raw}" in desc_l:
            if norm not in vol_days:
                violations.append(f"Does not match preferred day: {norm}")

    for block in VALID_TIME_BLOCKS:
        bl = block.lower()
        if f"prefer {bl}" in desc_l or f"ideally {bl}" in desc_l:
            if block not in vol_blocks:
                violations.append(f"Does not match preferred time block: {block}")

    return violations


def run_matching(
    need_set: dict,
    confirmed_skills: list,
    form_certs: list,
    form_languages: list,
    has_specific_date: bool,
    target_date_str: Optional[str],
    notification_date_str: str,
    is_recurring: bool,
    roster_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
) -> dict:
    """Execute deterministic volunteer matching against the roster.

    This is the most complex function in the system.  It:
    1. Builds the full set of hard requirements by merging form fields,
       classifier output, and organizational policy.
    2. Defines a check function per requirement, each mapped to its
       roster column name (for counterfactual reporting).
    3. Runs every volunteer through ALL checks.
    4. For unmatched volunteers, identifies which SINGLE requirement
       blocked them (counterfactual analysis).
    5. Computes capacity margins for all matched volunteers.
    """

    # ── Step 1: Merge all hard requirements ────────────────────────────

    required_skills = set(confirmed_skills)

    # Certs: explicit form selections + policy-mandated certs
    mandatory_certs = infer_mandatory_certs(confirmed_skills)
    required_certs = set(form_certs) | set(mandatory_certs)

    # Languages: form selections join AND; classifier may add OR logic.
    # Normalize with canonicalization and AND/OR deduplication.
    classifier_langs = need_set.get("languages", {"AND": [], "OR": []})
    merged_lang_and = list(set(form_languages) | set(classifier_langs.get("AND", [])))
    merged_lang_or = classifier_langs.get("OR", [])
    merged_lang_requirement = normalize_flexible_requirement(
        {"AND": merged_lang_and, "OR": merged_lang_or}, domain="languages"
    )

    # Availability days: from classifier, normalized
    days_requirement = normalize_flexible_requirement(
        need_set.get("availability_days", {"AND": [], "OR": []}), domain="days"
    )

    # Time blocks: canonicalized
    required_time_blocks = {
        canonicalize_value(v, "time_blocks")
        for v in need_set.get("availability_time_blocks", [])
        if canonicalize_value(v, "time_blocks")
    }

    required_hours = need_set.get("min_hours")
    required_transport = need_set.get("transportation_needed")

    # Notice period
    notice_days_available = None
    if has_specific_date and target_date_str:
        try:
            t_date = date.fromisoformat(target_date_str)
            n_date = date.fromisoformat(notification_date_str)
            notice_days_available = (t_date - n_date).days
        except (ValueError, TypeError):
            pass

    # ── Step 2: Define requirement checks ──────────────────────────────

    def check_active(vol_row):
        return str(vol_row["status"]).strip().casefold() == "active"

    def check_skills(vol_row):
        if not required_skills:
            return True
        vol_skills = parse_semicolon(vol_row["skills"], domain="skills")
        return required_skills.issubset(vol_skills)

    def check_certs(vol_row):
        if not required_certs:
            return True
        vol_certs = parse_semicolon(vol_row["certifications"])
        return required_certs.issubset(vol_certs)

    def check_languages(vol_row):
        if not merged_lang_requirement.get("AND") and not merged_lang_requirement.get("OR"):
            return True
        vol_langs = parse_semicolon(vol_row["languages"], domain="languages")
        return evaluate_flexible_requirement(merged_lang_requirement, vol_langs)

    def check_days(vol_row):
        if not days_requirement.get("AND") and not days_requirement.get("OR"):
            return True
        vol_days = parse_semicolon(vol_row["availability_days"], domain="days")
        return evaluate_flexible_requirement(days_requirement, vol_days)

    def check_time_blocks(vol_row):
        if not required_time_blocks:
            return True
        vol_blocks = parse_semicolon(
            vol_row["availability_time_blocks"], domain="time_blocks"
        )
        return required_time_blocks.issubset(vol_blocks)

    def check_notice(vol_row):
        if notice_days_available is None:
            return True
        vol_min_notice = vol_row["min_notice_days"]
        if pd.isna(vol_min_notice):
            return True
        return notice_days_available >= vol_min_notice

    def check_hours(vol_row):
        if required_hours is None:
            return True
        if not has_specific_date:
            max_hrs = vol_row["max_hours_per_week"]
            if pd.isna(max_hrs):
                return True
            return required_hours <= max_hrs
        committed = get_committed_hours(
            vol_row["volunteer_id"], target_date_str, assignments_df
        )
        max_hrs = vol_row["max_hours_per_week"]
        if pd.isna(max_hrs):
            return True
        return (committed + required_hours) <= max_hrs

    def check_transport(vol_row):
        if not required_transport or required_transport == "Any":
            return True
        vol_transport = canonicalize_value(
            vol_row.get("transportation", ""), "transportation"
        )
        req_transport = canonicalize_value(required_transport, "transportation")
        if req_transport == "Car":
            return vol_transport == "Car"
        return True

    requirements = [
        ("Active Status",           check_active,      "status"),
        ("Required Skills",         check_skills,      "skills"),
        ("Required Certifications", check_certs,       "certifications"),
        ("Language Requirements",   check_languages,   "languages"),
        ("Availability Days",       check_days,        "availability_days"),
        ("Time Block Fit",          check_time_blocks, "availability_time_blocks"),
        ("Notice Period",           check_notice,      "min_notice_days"),
        ("Hours Capacity",          check_hours,       "max_hours_per_week"),
        ("Transportation",          check_transport,   "transportation"),
    ]

    # ── Step 3: Run matching ───────────────────────────────────────────

    matched = []
    all_results = {}

    for _, vol in roster_df.iterrows():
        vid = vol["volunteer_id"]
        results = {}
        for req_name, check_fn, _ in requirements:
            results[req_name] = check_fn(vol)
        all_results[vid] = results

        if all(results.values()):
            matched.append(vid)

    # ── Step 4: Counterfactual analysis ────────────────────────────────

    counterfactuals = {}
    almost_matched_list = []

    for req_name, _, col_name in requirements:
        solely_blocked = []
        for _, vol in roster_df.iterrows():
            vid = vol["volunteer_id"]
            if vid in matched:
                continue

            res = all_results[vid]
            fails_this = not res[req_name]
            passes_all_others = all(
                res[other_name]
                for other_name, _, _ in requirements
                if other_name != req_name
            )

            if fails_this and passes_all_others:
                solely_blocked.append({
                    "volunteer_id": vid,
                    "preferred_name": vol["preferred_name"],
                    "blocking_requirement": req_name,
                    "blocking_column": col_name,
                })
                almost_matched_list.append({
                    "volunteer_id": vid,
                    "preferred_name": vol["preferred_name"],
                    "blocking_requirement": req_name,
                    "blocking_column": col_name,
                    "skills": vol["skills"],
                    "certifications": vol["certifications"],
                    "availability_days": vol["availability_days"],
                    "availability_time_blocks": vol["availability_time_blocks"],
                    "availability_notes": str(vol.get("availability_notes", "")),
                    "preferred_roles": str(vol.get("preferred_roles", "")),
                    "notes": str(vol.get("notes", "")),
                })

        if solely_blocked:
            counterfactuals[req_name] = solely_blocked

    # ── Step 5: Compute margins for matched volunteers ─────────────────

    margins = {}
    for vid in matched:
        vol = roster_df[roster_df["volunteer_id"] == vid].iloc[0]
        vol_skills = parse_semicolon(vol["skills"], domain="skills")
        vol_certs = parse_semicolon(vol["certifications"])
        vol_langs = parse_semicolon(vol["languages"])
        vol_days = parse_semicolon(vol["availability_days"])

        extra_skills = sorted(vol_skills - required_skills)
        extra_certs = sorted(vol_certs - required_certs)
        required_lang_values = get_all_required_values(merged_lang_requirement)
        extra_langs = sorted(vol_langs - required_lang_values)

        # Notice slack
        if notice_days_available is not None and not pd.isna(vol["min_notice_days"]):
            notice_slack = notice_days_available - vol["min_notice_days"]
        else:
            notice_slack = vol["min_notice_days"] if not pd.isna(vol["min_notice_days"]) else 0

        # Hours remaining
        max_hrs = vol["max_hours_per_week"] if not pd.isna(vol["max_hours_per_week"]) else 0
        committed = get_committed_hours(vid, target_date_str, assignments_df)
        hours_after_request = max_hrs - committed
        if required_hours:
            hours_after_request -= required_hours

        required_day_values = get_all_required_values(days_requirement)
        extra_days = sorted(vol_days - required_day_values)

        margins[vid] = {
            "extra_skills": extra_skills,
            "extra_certifications": extra_certs,
            "extra_languages": extra_langs,
            "notice_slack_days": notice_slack,
            "has_specific_date": has_specific_date,
            "vol_min_notice_days": vol["min_notice_days"] if not pd.isna(vol["min_notice_days"]) else 0,
            "hours_committed_this_week": round(committed, 1),
            "hours_remaining_after_request": round(hours_after_request, 1),
            "max_hours_per_week": max_hrs,
            "extra_availability_days": extra_days,
        }

    return sanitize_for_state({
        "matched": matched,
        "margins": margins,
        "counterfactuals": counterfactuals,
        "almost_matched": almost_matched_list,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — LANGGRAPH NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def classify_needs_node(state: GraphState) -> dict:
    """LLM node: extracts structured volunteer requirements from natural language.

    Uses with_structured_output to guarantee ClassifierOutput schema.
    KEY FEATURES:
    - From app_6: soft_preferences field in schema + signal-word prompt
    - From app_4: post-extraction vocabulary sanitization
    - From app_5: skill validation against VALID_SKILLS
    """
    llm = ChatOpenAI(
        model=st.session_state.get("model_name", "gpt-5-nano"),
        temperature=0
    )
    structured_llm = llm.with_structured_output(ClassifierOutput)

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

    result: ClassifierOutput = structured_llm.invoke([
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

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
            is_recurring=state.get("is_recurring", False),
            roster_df=available_roster,
            assignments_df=assignments,
        )

        ns_matched = match_result["matched"]
        all_matched.append({
            "need_set_index": ns_idx,
            "need_set_description": need_set.get("description", ""),
            "count_needed": need_set.get("count", 1),
            "matched_volunteer_ids": ns_matched,
        })

        all_margins.update(match_result["margins"])

        for req_name, blocked in match_result["counterfactuals"].items():
            key = f"NS{ns_idx}: {req_name}"
            all_counterfactuals[key] = blocked

        all_almost.extend(match_result["almost_matched"])

        for vid in ns_matched[:need_set.get("count", 1)]:
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


def recommend_node(state: GraphState) -> dict:
    """LLM node: tiers matched volunteers and generates configuration/gap advice.

    KEY DESIGN DECISIONS:
    1. Prompt compression: only send soft-fit-relevant fields to the LLM.
       Capacity metrics, history, and home area are displayed deterministically.
    2. Combined soft preferences: classifier's explicit field + unchecked skills.
    3. System prompt is purely instructional (from app_5); all data in human msg.
    4. Post-processing validates tiers (from app_6) and backfills Technical Match
       for omitted matched volunteers (from app_5).
    """
    roster = load_roster()

    # ── Build the context document ────────────────────────────────────
    sections = []

    sections.append("=== ORIGINAL REQUEST ===")
    sections.append(f"Prompt: {state['user_prompt']}")
    sections.append(f"Confirmed hard skills: {state['confirmed_skills']}")
    sections.append(f"Required certs: {state['form_certs']}")
    sections.append(f"Required languages: {state['form_languages']}")
    if state["has_specific_date"]:
        sections.append(f"Target date: {state['target_date']}")
    # ── Combined soft preferences ──────────────────────────────────────
    # Two sources: (1) classifier's explicit field, (2) unchecked skills.
    classifier_soft = state.get("soft_preferences", "")
    unchecked = state.get("unchecked_skills", [])

    has_soft = classifier_soft or unchecked
    if has_soft:
        sections.append("\n=== SOFT PREFERENCES (inform tiering, NOT hard filters) ===")
        if classifier_soft:
            sections.append(f"From request language: {classifier_soft}")
        if unchecked:
            sections.append(f"Suggested skills not confirmed as required: {unchecked}")
        sections.append(
            "Volunteers lacking these should be Good Match at best, never Perfect Match."
        )
    else:
        sections.append("\n=== SOFT PREFERENCES ===")
        sections.append(
            "No soft preferences identified. Tier based on hard requirements and fit only."
        )

    # ── Format MATCHED volunteers (compressed: only soft-fit fields) ───
    valid_matched_ids = set()
    valid_almost_ids = set()
    soft_flags_by_vid = {}  # For rule-based violation detection

    for match_group in state["matched_volunteers"]:
        ns_desc = match_group["need_set_description"]
        count_needed = match_group["count_needed"]
        sections.append(
            f"\n=== NEED SET: {ns_desc} (need {count_needed} volunteer(s)) ==="
        )

        for vid in match_group["matched_volunteer_ids"]:
            valid_matched_ids.add(vid)
            vol_rows = roster[roster["volunteer_id"] == vid]
            if vol_rows.empty:
                continue
            vol = vol_rows.iloc[0]

            # Rule-based soft-preference violation detection
            need_set = state["need_sets"][match_group["need_set_index"]]
            violations = summarize_soft_preference_violations(need_set, vol)
            if unchecked:
                vol_skills = parse_semicolon(vol["skills"])
                missing_soft = [s for s in unchecked if s not in vol_skills]
                if missing_soft:
                    violations.append(
                        f"Missing suggested (not required) skills: {missing_soft}"
                    )
            soft_flags_by_vid[vid] = violations

            # Compressed volunteer profile: only what the recommender needs
            violation_note = f"\n  ⚠ Soft-preference violations: {violations}" if violations else ""
            sections.append(f"""
VOLUNTEER: {vol['preferred_name']} ({vid}) — MATCHED{violation_note}
  Skills: {truncate_text(vol['skills'])}
  Preferred Roles: {truncate_text(vol.get('preferred_roles', ''))}
  Certifications: {truncate_text(vol['certifications'])}
  Availability: {vol['availability_days']} / {vol['availability_time_blocks']}
  Availability Notes: {truncate_text(vol.get('availability_notes', ''))}
  Transportation: {vol.get('transportation', 'N/A')}
  Languages: {vol['languages']}
  Notes: {truncate_text(vol.get('notes', ''))}""")

    # ── Format ALMOST-MATCHED volunteers ──────────────────────────────
    if state["almost_matched"]:
        sections.append(
            "\n=== ALMOST-MATCHED (blocked by 1 hard requirement) ==="
        )
        for am in state["almost_matched"]:
            vid = am["volunteer_id"]
            valid_almost_ids.add(vid)
            vol_rows = roster[roster["volunteer_id"] == vid]
            pref_roles = ""
            if not vol_rows.empty:
                pref_roles = truncate_text(vol_rows.iloc[0].get("preferred_roles", ""))

            sections.append(f"""
VOLUNTEER: {am['preferred_name']} ({vid}) — ALMOST MATCH
  BLOCKED BY: {am['blocking_requirement']} (column: {am['blocking_column']})
  Skills: {truncate_text(am.get('skills', ''))}
  Preferred Roles: {pref_roles}
  Certs: {truncate_text(am.get('certifications', ''))}
  Availability: {am.get('availability_days', '')} / {am.get('availability_time_blocks', '')}
  Notes: {truncate_text(am.get('notes', ''))}""")

    full_context = "\n".join(sections)

    # ── Call the LLM ──────────────────────────────────────────────────
    llm = ChatOpenAI(
        model=st.session_state.get("model_name", "gpt-5-nano"),
        temperature=0.2
    )
    structured_llm = llm.with_structured_output(RecommenderOutput)

    result: RecommenderOutput = structured_llm.invoke([
        SystemMessage(content=RECOMMENDER_SYSTEM_PROMPT),
        HumanMessage(content=full_context),
    ])

    # ── Full post-processing pipeline ─────────────────────────────────
    # Combines: tier enforcement (app_6), soft-preference demotion (app_5),
    # Technical Match backfill (app_5), hallucinated-ID filtering (app_4/6).

    all_valid_ids = valid_matched_ids | valid_almost_ids
    processed_recs = []
    seen_vids = set()

    for rec in result.recommendations:
        rec_dict = rec.model_dump()
        vid = rec_dict["volunteer_id"]

        # Drop hallucinated IDs
        if vid not in all_valid_ids:
            continue

        # Force almost-matched → Almost Match (from app_6)
        if vid in valid_almost_ids and vid not in valid_matched_ids:
            if rec_dict["tier"] != "Almost Match":
                rec_dict["tier"] = "Almost Match"
                rec_dict["reasoning"] = (
                    f"[Tier corrected — blocked by hard requirement.] "
                    f"{rec_dict['reasoning']}"
                )

        # Demote Perfect Match → Good Match if soft violations (from app_5)
        if (rec_dict["tier"] == "Perfect Match"
                and soft_flags_by_vid.get(vid)):
            rec_dict["tier"] = "Good Match"
            rec_dict["reasoning"] = (
                f"[Adjusted from Perfect — soft preference gap.] "
                f"{rec_dict['reasoning']}"
            )

        seen_vids.add(vid)
        processed_recs.append(rec_dict)

    # ── Backfill: any matched volunteer the LLM omitted → Technical Match ──
    # From app_5: ensures every matched volunteer appears in the output.
    for match_group in state["matched_volunteers"]:
        for vid in match_group["matched_volunteer_ids"]:
            if vid not in seen_vids:
                vol_rows = roster[roster["volunteer_id"] == vid]
                name = vol_rows.iloc[0]["preferred_name"] if not vol_rows.empty else vid
                processed_recs.append({
                    "volunteer_id": vid,
                    "tier": "Technical Match",
                    "reasoning": (
                        f"{name} passes all hard requirements but was not "
                        f"explicitly ranked by the recommender."
                    ),
                })
                seen_vids.add(vid)

    return sanitize_for_state({
        "recommendations": processed_recs,
        "configuration_notes": result.configuration_notes,
        "gap_notes": result.gap_notes,
    })


def write_request_record_node(state: GraphState) -> dict:
    """Terminal node: persists the full request record to the requests CSV."""
    record = {
        "request_id": str(uuid.uuid4())[:8],
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
        "has_specific_date": state["has_specific_date"],
        "target_date": state.get("target_date", ""),
        "notification_date": state["notification_date"],
        "is_recurring": state.get("is_recurring", False),
        "matched_volunteers_json": json_dumps_safe(state["matched_volunteers"]),
        "margins_json": json_dumps_safe(state["margins"]),
        "counterfactuals_json": json_dumps_safe(state["counterfactuals"]),
        "almost_matched_json": json_dumps_safe(state["almost_matched"]),
        "recommendations_json": json_dumps_safe(state["recommendations"]),
        "resulting_assignment_ids": "[]",
    }

    record_df = pd.DataFrame([record])
    if os.path.exists(REQUESTS_DATA_PATH):
        record_df.to_csv(REQUESTS_DATA_PATH, mode="a", header=False, index=False)
    else:
        record_df.to_csv(REQUESTS_DATA_PATH, index=False)

    return sanitize_for_state({"request_record": record})


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """Construct the LangGraph state-graph with human-in-the-loop interrupt.

    The graph pauses after classify_needs so the user can confirm which
    extracted skills should be treated as hard requirements.  When resumed,
    it runs match_volunteers → recommend → write_request_record.
    """
    builder = StateGraph(GraphState)

    builder.add_node("classify_needs", classify_needs_node)
    builder.add_node("match_volunteers", match_volunteers_node)
    builder.add_node("recommend", recommend_node)
    builder.add_node("write_request_record", write_request_record_node)

    builder.add_edge(START, "classify_needs")
    builder.add_edge("classify_needs", "match_volunteers")
    builder.add_edge("match_volunteers", "recommend")
    builder.add_edge("recommend", "write_request_record")
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

    # ── Sidebar: API configuration ─────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_name = st.selectbox(
            "Model",
            ["gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-4o-mini"],
            index=0,
            help="gpt-5-nano is fast and cheapest; gpt-5-mini for more complex extraction; "
                 "gpt-5 for highest quality; gpt-4o-mini as a legacy fallback.",
        )
        st.session_state["model_name"] = model_name

        st.divider()
        st.markdown("**Data Files**")
        st.caption(f"Roster: `{ROSTER_PATH}`")
        st.caption(f"Assignments: `{ASSIGNMENTS_PATH}`")
        st.caption(f"Requests log: `{REQUESTS_DATA_PATH}`")

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

    # Show what mandatory certs will be auto-added
    auto_certs = infer_mandatory_certs(confirmed)
    if auto_certs:
        st.caption(
            f"🔒 Auto-added certifications based on confirmed skills: "
            f"{', '.join(auto_certs)}"
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

    # ── Configuration / gap notes ──────────────────────────────────────
    if state.get("configuration_notes"):
        st.info(f"**Configuration notes:** {state['configuration_notes']}")

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

                # Recommender's reasoning
                st.caption(rec["reasoning"])

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
