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
from core.llm import (
    CLASSIFIER_MODEL, SCORER_MODEL, REASONING_MODEL,
    call_classifier, call_likert_item, call_reasoning,
)
from core.scoring import (
    LIKERT_ITEMS, SCORE_MAP, SCORING_UNAVAILABLE_NOTE, collapse_box,
    build_scorer_shared_context, build_volunteer_profile, run_scoring_waves,
    compute_cap_reasons, postprocess_recommendations, build_gap_notes,
)
from core.reasoning import build_reasoning_bundle, fetch_reasoning
from core.records import (
    REQUESTS_DB_PATH, SCHEMA_VERSION, db_connect, init_request_db,
    insert_request_record, log_reasoning_event,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
# Every enum list below was extracted directly from the roster CSV.  The
# classifier and matcher both reference these so that vocabulary is aligned
# end-to-end: if a value isn't in these lists, it can't appear in the roster
# and would silently fail matching.





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
