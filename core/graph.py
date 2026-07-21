"""LangGraph nodes and graph construction.

Phase 4 verbatim move out of app.py (the scoring node, SECTION 8, and
SECTION 9).  Zero behavior edits; cross-module references are
import-qualified only.
"""

import uuid
from datetime import datetime

# LangGraph manages the stateful orchestration graph.  StateGraph passes
# typed state between nodes; InMemorySaver checkpoints state so we can
# interrupt (for human-in-the-loop skills confirmation) and resume.
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from core import llm, matching, policy, records, schemas, scoring
from core.schemas import GraphState

def score_volunteers_node(state: GraphState) -> dict:
    """LLM node: four Likert items per matched volunteer, in waves.

    Tier assignment is deterministic: threshold mapping over the summed
    item scores, then caps, then ordering — all in
    postprocess_recommendations.  Raw selections, boxes, and totals ride
    along so the record never discards the distribution.  Almost-matched
    volunteers never enter the scorer (structural, not policy).
    """
    roster = matching.load_roster()

    units = []
    caps_by_vid = {}
    names_by_vid = {}
    seen_vids = set()
    for match_group in state["matched_volunteers"]:
        need_set = state["need_sets"][match_group["need_set_index"]]
        shared_ctx = scoring.build_scorer_shared_context(state, need_set)
        for vid in match_group["matched_volunteer_ids"]:
            if vid in seen_vids:
                continue
            seen_vids.add(vid)
            vol_rows = roster[roster["volunteer_id"] == vid]
            if vol_rows.empty:
                continue
            vol = vol_rows.iloc[0]
            caps_by_vid[vid] = scoring.compute_cap_reasons(state, need_set, vol)
            names_by_vid[vid] = vol["preferred_name"]
            units.append({
                "volunteer_id": vid,
                "shared_ctx": shared_ctx,
                "profile": scoring.build_volunteer_profile(vol),
            })

    scores = scoring.run_scoring_waves(units)

    recs = []
    for unit in units:
        vid = unit["volunteer_id"]
        score = scores[vid]
        if score.get("failed"):
            recs.append({
                "volunteer_id": vid,
                "tier": "Technical Match",
                "reasoning": scoring.SCORING_UNAVAILABLE_NOTE,
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
            "boxes": [scoring.collapse_box(s) for s in selections],
            "total_score": sum(scoring.SCORE_MAP[s] for s in selections),
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

    recs = scoring.postprocess_recommendations(recs, caps_by_vid, names_by_vid)

    return schemas.sanitize_for_state({
        "recommendations": recs,
        "gap_notes": scoring.build_gap_notes(
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

    result = llm.call_classifier(user_msg)

    # ── Collect and validate extracted skills ────────────────────────────
    # Fix 10: canonicalize before membership testing — a classifier emitting
    # "tutoring - math" must not be silently dropped from the confirmation
    # UI (never confirmable → never enforced → never cert-triggering).
    all_skills = []
    seen = set()
    for ns in result.need_sets:
        for sk in ns.applicable_skills:
            csk = matching.canonicalize_value(sk, "skills")
            if csk and csk not in seen and csk in policy.VALID_SKILLS:
                all_skills.append(csk)
                seen.add(csk)

    # ── Sanitize ALL vocabulary fields in each need set (from app_4) ─────
    # The LLM can produce near-miss strings like "spanish" instead of
    # "Spanish" or "monday" instead of "Mon" — these silently fail matching
    # because Python string comparison is case-sensitive.  We validate and
    # canonicalize every field.
    valid_langs_set = set(policy.VALID_LANGUAGES)
    valid_days_set = set(policy.VALID_DAYS)
    valid_time_set = set(policy.VALID_TIME_BLOCKS)
    valid_areas_set = set(policy.VALID_AREAS)

    sanitized_need_sets = []
    for ns in result.need_sets:
        ns_dict = ns.model_dump()

        # Validate languages in FlexibleRequirement
        lang_req = ns_dict.get("languages", {"AND": [], "OR": []})
        lang_req["AND"] = [
            matching.canonicalize_value(v, "languages")
            for v in lang_req.get("AND", [])
            if matching.canonicalize_value(v, "languages") in valid_langs_set
        ]
        lang_req["OR"] = [
            [matching.canonicalize_value(v, "languages") for v in branch
             if matching.canonicalize_value(v, "languages") in valid_langs_set]
            for branch in lang_req.get("OR", [])
        ]
        lang_req["OR"] = [b for b in lang_req["OR"] if b]
        ns_dict["languages"] = lang_req

        # Validate days in FlexibleRequirement
        days_req = ns_dict.get("availability_days", {"AND": [], "OR": []})
        days_req["AND"] = [
            matching.canonicalize_value(v, "days")
            for v in days_req.get("AND", [])
            if matching.canonicalize_value(v, "days") in valid_days_set
        ]
        days_req["OR"] = [
            [matching.canonicalize_value(v, "days") for v in branch
             if matching.canonicalize_value(v, "days") in valid_days_set]
            for branch in days_req.get("OR", [])
        ]
        days_req["OR"] = [b for b in days_req["OR"] if b]
        ns_dict["availability_days"] = days_req

        # Validate time blocks
        ns_dict["availability_time_blocks"] = [
            matching.canonicalize_value(v, "time_blocks")
            for v in ns_dict.get("availability_time_blocks", [])
            if matching.canonicalize_value(v, "time_blocks") in valid_time_set
        ]

        # Validate location area
        if ns_dict.get("location_area") and ns_dict["location_area"] not in valid_areas_set:
            ns_dict["location_area"] = None

        # Validate applicable_skills (fix 10: canonicalize like other domains)
        canonical_skills = []
        for s in ns_dict.get("applicable_skills", []):
            cs = matching.canonicalize_value(s, "skills")
            if cs in policy.VALID_SKILLS and cs not in canonical_skills:
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
    roster = matching.load_roster()
    assignments = matching.load_assignments()

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
        pool_result = matching.run_matching(
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

        match_result = matching.run_matching(
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
        all_histories[vid] = matching.compute_volunteer_history(vid, assignments)

    return schemas.sanitize_for_state({
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
        "schema_version": records.SCHEMA_VERSION,
        "timestamp": datetime.now().isoformat(),
        "user_prompt": state["user_prompt"],
        "soft_preferences": state.get("soft_preferences", ""),
        "unchecked_skills": schemas.json_dumps_safe(state.get("unchecked_skills", [])),
        "request_source": "user_input",
        "need_sets_json": schemas.json_dumps_safe(state["need_sets"]),
        "confirmed_skills_json": schemas.json_dumps_safe(state["confirmed_skills"]),
        "extracted_skills_json": schemas.json_dumps_safe(state["extracted_skills"]),
        "form_certs_json": schemas.json_dumps_safe(state["form_certs"]),
        "form_languages_json": schemas.json_dumps_safe(state["form_languages"]),
        "has_specific_date": bool(state["has_specific_date"]),
        "target_date": state.get("target_date", ""),
        "notification_date": state["notification_date"],
        "is_recurring": bool(state.get("is_recurring", False)),
        "matched_volunteers_json": schemas.json_dumps_safe(state["matched_volunteers"]),
        "margins_json": schemas.json_dumps_safe(state["margins"]),
        "counterfactuals_json": schemas.json_dumps_safe(state["counterfactuals"]),
        "almost_matched_json": schemas.json_dumps_safe(state["almost_matched"]),
        "recommendations_json": schemas.json_dumps_safe(state["recommendations"]),
        "gap_notes": state.get("gap_notes") or "",
        "resulting_assignment_ids": "[]",
    }

    records.init_request_db()
    records.insert_request_record(record)

    return schemas.sanitize_for_state({"request_record": record})


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


