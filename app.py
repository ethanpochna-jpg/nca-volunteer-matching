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
# app.py is UI only (Phase 4): every schema, constant, matcher rule, LLM
# helper, scoring stage, and persistence path lives in core/ — this file
# renders the three Streamlit stages and calls through the modules.

import os                       # File-existence checks for data files
import uuid                     # Unique IDs for graph threads
from datetime import date, timedelta  # Form defaults and the date guard

import streamlit as st          # UI framework for the multi-step form interface

# Load .env so ANTHROPIC_API_KEY is available without hardcoding it.
from dotenv import load_dotenv
load_dotenv()

from core import graph, llm, matching, policy, reasoning, records, scoring


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — STREAMLIT USER INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

# §12 aesthetics: the ONLY raw-CSS surface in the app. Everything themable
# stays in .streamlit/config.toml; these six rules exist solely because
# per-element styling (tier accent bars on keyed containers, container
# padding, page gutter) has no [theme] key. Selectors hang off Streamlit's
# documented stable `st-key-<key>` classes from st.container(key=...).
_BRAND_CSS = """
<style>
div[class*="st-key-card-perfect-"]   { border-left: 4px solid #2E7D32; background: #F4FAF5; }
div[class*="st-key-card-good-"]      { border-left: 4px solid #2563EB; background: #F3F7FE; }
div[class*="st-key-card-technical-"] { border-left: 4px solid #B45309; background: #FDF9F1; }
div[class*="st-key-card-almost-"]    { border-left: 4px solid #BE123C; background: #FDF3F5; }
div[class*="st-key-card-"]           { padding: 1rem 1.25rem 1.1rem; }
div.block-container                  { padding-top: 2.2rem; }
</style>
"""


def inject_brand_css() -> None:
    """Inject the §12 brand CSS once per rerun, right after page config."""
    st.markdown(_BRAND_CSS, unsafe_allow_html=True)


# §12 score chips: raw 1–5 selections rendered as theme-palette badges.
# Box→color mirrors the collapse semantics (T2B good / Neutral flat /
# B2B bad) without the model or the UI ever re-deciding a tier.
_CHIP_LABELS = {
    "overall_fit": "Fit",
    "schedule_friction": "Schedule",
    "willingness": "Willing",
    "recommendation": "Recommend",
}
_BOX_BADGE = {"T2B": "green", "Neutral": "gray", "B2B": "orange"}


def format_score_chips(rec: dict) -> str:
    """Markdown badge chips for one recommendation card.

    Pure function so the chip contract is unit-testable: Almost Match recs
    (no raw_selections key) get no chips; a failed scorer (raw_selections
    is None) gets the policy-fallback badge, never fake scores.
    """
    if "raw_selections" not in rec:
        return ""
    if rec.get("raw_selections") is None:
        return ":gray-badge[Scoring unavailable — Technical by policy]"
    chips = [
        f":{_BOX_BADGE[box]}-badge[{_CHIP_LABELS[item['key']]} {sel}]"
        for item, sel, box in zip(
            scoring.LIKERT_ITEMS, rec["raw_selections"], rec["boxes"]
        )
    ]
    chips.append(f":blue-badge[Total {rec['total_score']:+d}]")
    return " ".join(chips)


def format_dissent_badge(event: dict) -> str | None:
    """§12 visual dissent marker for a fetched reasoning event.

    Violet on purpose — distinct from both the B2B-orange chips and the
    Almost-Match red accent. The tier itself never moves (I5/D-G); this
    badge only surfaces that the logged event carried dissent=1.
    """
    if event.get("dissent"):
        return ":violet-badge[⚑ Dissent — reasoning disagrees; tier unchanged]"
    return None


def main():
    """Entry point: page config, sidebar, session state, stage dispatch."""

    st.set_page_config(
        page_title="NCA Volunteer Matching Assistant",
        page_icon="🤝",
        layout="wide",
    )
    inject_brand_css()

    st.title("🤝 Northbridge Volunteer Matching Assistant")
    st.caption(
        "AI-powered volunteer coordination for program managers. "
        "Describe what you need — the system identifies, filters, and recommends volunteers."
    )

    # ── Sidebar: read-only configuration (D-J: no model selector) ──────
    with st.sidebar:
        st.subheader("Configuration")
        st.markdown("**Models in use**")
        st.caption(f"Classifier: `{llm.CLASSIFIER_MODEL}`")
        st.caption(f"Scorer: `{llm.SCORER_MODEL}`")
        st.caption(f"Reasoning: `{llm.REASONING_MODEL}`")

        st.divider()
        st.markdown("**Data Files**")
        st.caption(f"Roster: `{matching.ROSTER_PATH}`")
        st.caption(f"Assignments: `{matching.ASSIGNMENTS_PATH}`")
        st.caption(f"Requests DB: `{records.REQUESTS_DB_PATH}`")
        st.badge("Demo dataset — resets on redeploy", color="gray")

    # ── Session state initialization ───────────────────────────────────
    if "stage" not in st.session_state:
        st.session_state["stage"] = "input"
    if "graph" not in st.session_state:
        st.session_state["graph"] = graph.build_graph()
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())

    # ── Verify data files exist ────────────────────────────────────────
    if not os.path.exists(matching.ROSTER_PATH):
        st.error(f"Roster file not found: `{matching.ROSTER_PATH}`. Place it in the app directory.")
        st.stop()
    if not os.path.exists(matching.ASSIGNMENTS_PATH):
        st.error(
            f"Assignments file not found: `{matching.ASSIGNMENTS_PATH}`. "
            f"Place it in the app directory."
        )
        st.stop()

    # ── Seed the demo request history on first run (S7) ────────────────
    # requests.db is generated and gitignored; a fresh deploy (or reboot
    # of the ephemeral container) rebuilds it from the seed script.
    if not os.path.exists(records.REQUESTS_DB_PATH):
        from data.seed_requests import seed_database
        records.init_request_db()
        with records.db_connect() as conn:
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

    st.subheader("📋 Describe Your Volunteer Need", divider="gray")

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

    st.subheader("📌 Hard Requirements", divider="gray")
    st.caption(
        "These override and supplement the natural language extraction. "
        "Anything selected here is treated as a non-negotiable filter."
    )

    # §12: bordered group — pure layout, widgets and params unchanged.
    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            form_certs = st.multiselect(
                "Required Certifications",
                options=policy.VALID_CERTS_CLEARABLE,
                help="Only volunteers with ALL selected certifications will match. "
                     "Policy-mandated certs (e.g., Background Check for tutoring) "
                     "are added automatically based on skills.",
            )

        with col2:
            form_languages = st.multiselect(
                "Required Languages",
                options=policy.VALID_LANGUAGES,
                help="Only volunteers who speak ALL selected languages will match.",
            )

    st.subheader("📅 Scheduling", divider="gray")
    # §12: bordered group — pure layout, widgets and params unchanged.
    with st.container(border=True):
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

        # ── Recurring toggle ───────────────────────────────────────────
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

    st.subheader("🔍 Review Extracted Requirements", divider="gray")

    # Show classifier reasoning
    with st.expander("Classifier Reasoning", expanded=True, icon=":material/psychology:"):
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
    with st.expander("Need Sets", expanded=True, icon=":material/inventory_2:"):
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
    st.subheader("✅ Confirm Required Skills", divider="gray")
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
        # §12: bordered group — pure layout, checkbox keys unchanged.
        with st.container(border=True):
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
    auto_certs = policy.infer_mandatory_certs(sorted(set(extracted) | set(confirmed)))
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
            st.session_state["graph"] = graph.build_graph()
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
    roster = matching.load_roster()

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
    # §12: slug feeds the keyed-container CSS accent (st-key-card-<slug>-*);
    # accent hexes live only in _BRAND_CSS so color stays a styling concern.
    TIER_STYLE = {
        "Perfect Match": {"slug": "perfect", "icon": "🌟", "badge": "green"},
        "Good Match": {"slug": "good", "icon": "👍", "badge": "blue"},
        "Technical Match": {"slug": "technical", "icon": "⚙️", "badge": "orange"},
        "Almost Match": {"slug": "almost", "icon": "⚠️", "badge": "red"},
    }

    for tier in tier_order:
        tier_recs = [r for r in recommendations if r["tier"] == tier]
        if not tier_recs:
            continue

        st.markdown(f"### {TIER_STYLE[tier]['icon']} {tier} ({len(tier_recs)})")

        for rec in tier_recs:
            vid = rec["volunteer_id"]
            vol_rows = roster[roster["volunteer_id"] == vid]
            if vol_rows.empty:
                continue
            vol = vol_rows.iloc[0]
            margin = state.get("margins", {}).get(vid, {})
            history = state.get("volunteer_histories", {}).get(vid, {})

            # §12: a real bordered card — the key's tier slug picks up the
            # accent bar from _BRAND_CSS (the old raw-<div> strip never
            # wrapped Streamlit children and is gone).
            with st.container(border=True, key=f"card-{TIER_STYLE[tier]['slug']}-{vid}"):
                # Header: name, ID, pronouns (with NaN guard)
                pronouns = vol.get("pronouns", "")
                pronouns_str = f" · {pronouns}" if pronouns and str(pronouns) not in ("nan", "", "NaN") else ""
                st.markdown(f"**{vol['preferred_name']}** ({vid}){pronouns_str}")

                # §12: per-item score chips (raw distribution, stored in the
                # record — displayed, never re-tiered here).
                chips = format_score_chips(rec)
                if chips:
                    st.markdown(chips)

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
                        dissent_badge = format_dissent_badge(cached)
                        if dissent_badge:
                            st.markdown(dissent_badge)
                        st.caption(cached["text"])
                    elif rec.get("reasoning"):
                        st.caption(rec["reasoning"])  # scoring-unavailable note

                    if st.button("💬 Get reasoning", key=f"reason_{vid}"):
                        bundle = reasoning.build_reasoning_bundle(
                            state.get("user_prompt", ""),
                            ns_desc_by_vid.get(vid, ""),
                            state.get("soft_preferences", ""),
                            vol,
                            rec,
                        )
                        fresh = cache_key not in cache
                        with st.spinner("Fetching reasoning..."):
                            event = reasoning.fetch_reasoning(bundle, tier, cache, cache_key)
                        # S7: one reasoning_events row per button FETCH
                        # (reruns replay from cache and log nothing).
                        request_id = state.get("request_record", {}).get("request_id")
                        if fresh and request_id:
                            records.log_reasoning_event(request_id, vid, event)
                        st.rerun()

                # ── Volunteer details in 3 columns ─────────────────
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("**Availability**")
                    # §12: one markdown block per column (soft line breaks)
                    # instead of stacked st.text — kills inter-line gaps.
                    c1_lines = [
                        f"Days: {vol['availability_days']}",
                        f"Time: {vol['availability_time_blocks']}",
                    ]
                    avail_notes = vol.get("availability_notes", "")
                    if avail_notes and str(avail_notes) not in ("nan", "", "NA"):
                        c1_lines.append(f"Notes: {avail_notes}")
                    st.markdown("  \n".join(c1_lines))

                with c2:
                    st.markdown("**Profile**")
                    transport = vol.get("transportation", "N/A")
                    c2_lines = [
                        f"Area: {vol['home_area']}",
                        f"Transport: {transport if str(transport) != 'nan' else 'N/A'}",
                        f"Languages: {vol['languages']}",
                    ]
                    pref_roles = vol.get("preferred_roles", "")
                    if pref_roles and str(pref_roles) not in ("nan", "", "NA"):
                        c2_lines.append(f"Preferred roles: {pref_roles}")
                    notes = vol.get("notes", "")
                    if notes and str(notes) not in ("nan", "", "NA"):
                        c2_lines.append(f"Notes: {notes}")
                    st.markdown("  \n".join(c2_lines))

                with c3:
                    # ── Capacity Margins ────────────────────────────
                    if tier != "Almost Match" and margin:
                        st.markdown("**Capacity Margins**")

                        hrs_rem = margin.get("hours_remaining_after_request")
                        max_hrs = margin.get("max_hours_per_week", "?")
                        committed = margin.get("hours_committed_this_week", 0)

                        m_lines = []
                        if margin.get("has_specific_date"):
                            if hrs_rem is not None:
                                m_lines.append(
                                    f"Hours remaining: {hrs_rem} "
                                    f"(of {max_hrs}/wk, {committed} committed)"
                                )
                            else:
                                m_lines.append(f"Max hours/wk: {max_hrs}")
                            notice_slack = margin.get("notice_slack_days")
                            if notice_slack is not None:
                                m_lines.append(f"Notice slack: {notice_slack} days")
                        else:
                            m_lines.append(f"Max hours/wk: {max_hrs}")
                            vol_notice = margin.get("vol_min_notice_days", "?")
                            m_lines.append(f"Min notice required: {vol_notice} days")

                        extra_skills = margin.get("extra_skills", [])
                        if extra_skills:
                            m_lines.append(f"Extra skills: {', '.join(extra_skills)}")
                        st.markdown("  \n".join(m_lines))

                    # ── Assignment History ──────────────────────────
                    st.markdown("**History**")
                    total_asgn = history.get("total_assignments", 0)
                    no_shows = history.get("no_shows", 0)
                    no_show_rate = history.get("no_show_rate", 0)
                    last_asgn = history.get("last_assigned", "No prior assignments")
                    st.markdown(
                        f"Assignments: {total_asgn}  \n"
                        f"No-shows: {no_shows} ({no_show_rate:.0%})  \n"
                        f"Last assigned: {last_asgn}"
                    )

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
        st.session_state["graph"] = graph.build_graph()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
