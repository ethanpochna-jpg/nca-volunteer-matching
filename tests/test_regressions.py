"""Regression suite — one test group per PLAN item, in landing order.

Layout:
  Phase 0  — harness smoke
  Phase 1  — fixes 2, 10, 12, 1, 3, 5, 6, 4 (dependency order)
  Phase 2  — S1 transport/migration, S2 constants, S3 scoring node,
             S4 tiering/caps, S5 assembly/gap notes, S6 reasoning/dissent,
             S7 SQLite persistence + seed
  Phase 3  — permanent guards (committed-hours ISO math, degenerate
             branches, per-need-set scoping, sanitize_for_state)

No test calls a live API: LLM configuration is asserted via
mocked-transport request bodies (conftest.make_mock_anthropic), and node
logic via monkeypatched call helpers.
"""

import json
import sqlite3 as _sqlite3
import uuid as _uuid

import pandas as pd

from tests.conftest import patch_llm, patch_loaders
from tests.fixtures import (
    assignments_frame,
    make_need_set,
    make_state,
    make_volunteer,
    roster_frame,
    run_matching_defaults,
)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0 — harness smoke
# ═══════════════════════════════════════════════════════════════════════════

class TestHarnessSmoke:
    def test_app_imports_without_streamlit_runtime(self, app):
        """Import executes only definitions — no UI, no network, no key."""
        for symbol in (
            "run_matching", "normalize_flexible_requirement",
            "infer_mandatory_certs", "match_volunteers_node",
            "get_committed_hours", "sanitize_for_state",
            "load_roster", "load_assignments",
        ):
            assert hasattr(app, symbol), f"app.py lost expected symbol {symbol}"

    def test_matcher_runs_on_fixture_frames(self, app):
        """End-to-end sanity: an unconstrained need set matches everyone."""
        roster = roster_frame(
            make_volunteer("V-0001", "Ada"),
            make_volunteer("V-0002", "Grace"),
        )
        result = run_matching_defaults(app, make_need_set(), roster)
        assert result["matched"] == ["V-0001", "V-0002"]
        assert result["almost_matched"] == []


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — fix 2: OR-branch subsumption vacates the whole OR clause
# ═══════════════════════════════════════════════════════════════════════════

class TestFix2OrBranchSubsumption:
    def test_and_implied_branch_vacates_whole_or(self, app):
        """AND=[Mon], OR=[[Mon],[Mon,Sat]] ≡ "Monday" — the executed proof.

        The [Mon] branch is satisfied whenever AND holds, so the OR clause
        adds nothing; the old code dropped only that branch and left
        [Sat] mandatory, rejecting Monday-only volunteers.
        """
        result = app.normalize_flexible_requirement(
            {"AND": ["Mon"], "OR": [["Mon"], ["Mon", "Sat"]]}, domain="days"
        )
        assert result == {"AND": ["Mon"], "OR": []}

    def test_monday_only_volunteer_passes_subsumed_requirement(self, app):
        """Matcher-level proof: Monday-only volunteer matches the ≡Monday req."""
        roster = roster_frame(
            make_volunteer("V-0001", "MondayOnly", availability_days="Mon"),
        )
        ns = make_need_set(
            description="Monday session",
            availability_days={"AND": ["Mon"], "OR": [["Mon"], ["Mon", "Sat"]]},
        )
        result = run_matching_defaults(app, ns, roster)
        assert result["matched"] == ["V-0001"]

    def test_garbage_only_branch_drops_without_vacating_sibling(self, app):
        """[""], ["NA"] branches are noise; the real [Sat] branch survives."""
        result = app.normalize_flexible_requirement(
            {"AND": [], "OR": [[""], ["NA"], ["Sat"]]}, domain="days"
        )
        assert result == {"AND": [], "OR": [["Sat"]]}

    def test_plain_cases_unchanged(self, app):
        """No subsumption, no garbage — canonicalization only."""
        result = app.normalize_flexible_requirement(
            {"AND": ["monday"], "OR": [["saturday"], ["sunday", "tuesday"]]},
            domain="days",
        )
        assert result == {"AND": ["Mon"], "OR": [["Sat"], ["Sun", "Tue"]]}

    def test_partial_subsumption_still_trims_branch(self, app):
        """A branch only PARTLY covered by AND keeps its remainder."""
        result = app.normalize_flexible_requirement(
            {"AND": ["Mon"], "OR": [["Mon", "Sat"], ["Sun"]]}, domain="days"
        )
        assert result == {"AND": ["Mon"], "OR": [["Sat"], ["Sun"]]}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — fix 10: fuzzy canonicalization for the skills domain
# ═══════════════════════════════════════════════════════════════════════════

class TestFix10SkillsCanonicalization:
    def test_lowercase_skill_survives_into_extracted_skills(self, app, monkeypatch):
        """'tutoring - math' from the classifier must reach the review UI.

        The mocked transport replaces only the network hop — the REAL
        post-LLM sanitization code runs.
        """
        canned = app.ClassifierOutput(
            need_sets=[app.NeedSet(
                count=1,
                description="Math tutor",
                applicable_skills=["tutoring - math", "Tutoring - Math"],
            )],
            reasoning="canned",
        ).model_dump_json()
        patch_llm(monkeypatch, app, [], [canned])
        out = app.classify_needs_node(make_state(user_prompt="math tutor"))
        assert out["extracted_skills"] == ["Tutoring - Math"]
        assert out["need_sets"][0]["applicable_skills"] == ["Tutoring - Math"]

    def test_roster_side_case_mismatch_matches(self, app):
        """Roster 'PANTRY OPERATIONS' satisfies required 'Pantry Operations'.

        (Pantry Operations triggers the food-handling cert rule, so the
        fixture volunteer must hold Food Safety to isolate the skills check.)
        """
        roster = roster_frame(
            make_volunteer(
                "V-0001", "Shouty",
                skills="PANTRY OPERATIONS",
                certifications="Food Safety - Basic",
            ),
            make_volunteer("V-0002", "NoSkills",
                           certifications="Food Safety - Basic"),
        )
        result = run_matching_defaults(
            app, make_need_set(), roster,
            confirmed_skills=["Pantry Operations"],
        )
        assert result["matched"] == ["V-0001"]

    def test_margins_report_canonical_extra_skills(self, app):
        """Extra-skill margins compare canonical forms, not raw strings."""
        roster = roster_frame(
            make_volunteer(
                "V-0001", "Multi",
                skills="PANTRY OPERATIONS; driver",
                certifications="Food Safety - Basic",
            ),
        )
        result = run_matching_defaults(
            app, make_need_set(), roster,
            confirmed_skills=["Pantry Operations"],
        )
        assert result["margins"]["V-0001"]["extra_skills"] == ["Driver"]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — fix 12: is_recurring dropped from the matcher signature
# ═══════════════════════════════════════════════════════════════════════════

class TestFix12IsRecurringRemoval:
    def test_run_matching_signature_lacks_is_recurring(self, app):
        """The parameter was never read in the body — dead weight removed.

        Its real use point arrives in Phase 2 (S3 scorer context line).
        """
        import inspect
        params = inspect.signature(app.run_matching).parameters
        assert "is_recurring" not in params
        assert list(params) == [
            "need_set", "confirmed_skills", "form_certs", "form_languages",
            "has_specific_date", "target_date_str", "notification_date_str",
            "roster_df", "assignments_df",
        ]

    def test_field_survives_in_state_and_record_paths(self, app):
        """Form/state/record keep the flag (D4) — only the matcher lost it."""
        assert "is_recurring" in app.GraphState.__annotations__
        import inspect
        record_src = inspect.getsource(app.write_request_record_node)
        assert "is_recurring" in record_src


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — fix 1: mandatory certs derive from work type, not checkboxes
# ═══════════════════════════════════════════════════════════════════════════

class TestFix1CertsFromWorkType:
    def _tutoring_roster(self):
        return roster_frame(
            make_volunteer(
                "V-NATE", "NoCheck Nate",
                skills="Tutoring - Math",
                certifications="",
            ),
            make_volunteer(
                "V-CLARA", "Cleared Clara",
                skills="Tutoring - Math",
                certifications=(
                    "Background Check - Cleared;"
                    "Child Safety Training - Completed"
                ),
            ),
        )

    def test_zero_confirmed_skills_still_requires_youth_certs(self, app):
        """The executed audit case: unchecked boxes must not disable policy."""
        ns = make_need_set(
            description="Math tutor for kids",
            applicable_skills=["Tutoring - Math"],
        )
        result = run_matching_defaults(
            app, ns, self._tutoring_roster(), confirmed_skills=[],
        )
        assert result["matched"] == ["V-CLARA"]
        nate_blocks = [
            am for am in result["almost_matched"]
            if am["volunteer_id"] == "V-NATE"
        ]
        assert nate_blocks
        assert nate_blocks[0]["blocking_requirement"] == "Required Certifications"

    def test_confirmed_skills_outside_need_set_also_trigger(self, app):
        """Union semantics: confirmed ∪ applicable, not intersection."""
        ns = make_need_set(description="General help", applicable_skills=[])
        result = run_matching_defaults(
            app, ns, self._tutoring_roster(),
            confirmed_skills=["Tutoring - Math"],
        )
        assert result["matched"] == ["V-CLARA"]

    def test_no_work_type_means_no_policy_certs(self, app):
        """Neutral work stays neutral — Nate matches when nothing triggers."""
        ns = make_need_set(description="Event help", applicable_skills=["Event Support"])
        result = run_matching_defaults(
            app, ns, self._tutoring_roster(), confirmed_skills=[],
        )
        assert set(result["matched"]) == {"V-NATE", "V-CLARA"}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — fix 3: scarcity-aware claiming across need sets
# ═══════════════════════════════════════════════════════════════════════════

class TestFix3ScarcityAwareClaiming:
    def test_two_slot_request_fills_both_slots(self, app, monkeypatch):
        """The executed proof: Ana (Spanish+Car) must be left for the
        driver slot; Bea (Spanish only) covers the intake slot."""
        # Ana precedes Bea in roster order — the old greedy claim took Ana
        # for the Spanish slot and left the driver slot unfillable.
        roster = roster_frame(
            make_volunteer("V-ANA", "Ana", languages="Spanish;English",
                           transportation="Car"),
            make_volunteer("V-BEA", "Bea", languages="Spanish;English",
                           transportation="Public Transit"),
        )
        patch_loaders(monkeypatch, app, roster, assignments_frame())
        state = make_state(need_sets=[
            make_need_set(description="Spanish-speaking intake volunteer",
                          languages={"AND": ["Spanish"], "OR": []}),
            make_need_set(description="Delivery driver",
                          transportation_needed="Car"),
        ])
        out = app.match_volunteers_node(state)
        intake, driver = out["matched_volunteers"]
        assert set(intake["matched_volunteer_ids"]) == {"V-ANA", "V-BEA"}
        assert driver["matched_volunteer_ids"] == ["V-ANA"]

    def test_tie_claims_in_pool_order(self, app, monkeypatch):
        """Equal scarcity → first-in-pool claimed, deterministically."""
        roster = roster_frame(
            make_volunteer("V-0001", "First"),
            make_volunteer("V-0002", "Second"),
        )
        patch_loaders(monkeypatch, app, roster, assignments_frame())
        state = make_state(need_sets=[
            make_need_set(description="Helper A"),
            make_need_set(description="Helper B"),
        ])
        out = app.match_volunteers_node(state)
        first_ns, second_ns = out["matched_volunteers"]
        assert first_ns["matched_volunteer_ids"] == ["V-0001", "V-0002"]
        # V-0001 claimed by NS0 (pool-order tie-break) → NS1 sees only V-0002
        assert second_ns["matched_volunteer_ids"] == ["V-0002"]

    def test_single_need_set_behavior_unchanged(self, app, monkeypatch):
        """One need set ⇒ scarcity is always zero ⇒ old behavior exactly."""
        roster = roster_frame(
            make_volunteer("V-0001", "A"),
            make_volunteer("V-0002", "B"),
            make_volunteer("V-0003", "C"),
        )
        patch_loaders(monkeypatch, app, roster, assignments_frame())
        state = make_state(need_sets=[make_need_set(count=2)])
        out = app.match_volunteers_node(state)
        assert out["matched_volunteers"][0]["matched_volunteer_ids"] == [
            "V-0001", "V-0002", "V-0003"
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — fix 5: per-group margins, first-wins flat view
# ═══════════════════════════════════════════════════════════════════════════

class TestFix5MarginStorage:
    def _dual_match_output(self, app, monkeypatch):
        """Cai (Spanish+English) matches BOTH need sets.

        Dev (Spanish only) is exclusive to the Spanish pool, so scarcity
        claiming takes Dev for the Spanish slot and Cai's margins appear
        under both need sets — the collision fix 5 addresses.
        """
        roster = roster_frame(
            make_volunteer("V-CAI", "Cai", languages="Spanish;English"),
            make_volunteer("V-DEV", "Dev", languages="Spanish"),
            make_volunteer("V-ANA", "Ana", languages="English"),
        )
        patch_loaders(monkeypatch, app, roster, assignments_frame())
        state = make_state(need_sets=[
            make_need_set(description="Spanish intake",
                          languages={"AND": ["Spanish"], "OR": []}),
            make_need_set(description="English-speaking helper",
                          languages={"AND": ["English"], "OR": []}),
        ])
        return app.match_volunteers_node(state)

    def test_margins_nested_per_group(self, app, monkeypatch):
        out = self._dual_match_output(app, monkeypatch)
        spanish_ns, english_ns = out["matched_volunteers"]
        # Against the Spanish-requiring need set, Spanish is NOT extra.
        assert spanish_ns["margins"]["V-CAI"]["extra_languages"] == ["English"]
        # Against the English-requiring need set, Spanish IS extra.
        assert english_ns["margins"]["V-CAI"]["extra_languages"] == ["Spanish"]

    def test_flat_view_first_need_set_wins(self, app, monkeypatch):
        """The executed proof: the flat display margins must not report
        Spanish as extra for the volunteer matched on the Spanish slot."""
        out = self._dual_match_output(app, monkeypatch)
        assert out["margins"]["V-CAI"]["extra_languages"] == ["English"]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — fix 6: word-boundary soft-preference detector
# ═══════════════════════════════════════════════════════════════════════════

class TestFix6SoftPreferenceDetector:
    def _weekday_volunteer_row(self):
        """A volunteer available Tue only — violates any Monday preference."""
        return roster_frame(
            make_volunteer("V-0001", "TueOnly", availability_days="Tue"),
        ).iloc[0]

    def test_prefer_monday_fires_exactly_once(self, app):
        """Old substring logic hit both the 'mon' and 'monday' aliases."""
        ns = make_need_set(description="We prefer monday sessions")
        violations = app.summarize_soft_preference_violations(
            ns, self._weekday_volunteer_row()
        )
        assert violations == ["Does not match preferred day: Mon"]

    def test_prefer_monetary_donations_no_false_positive(self, app):
        """'monetary' must not trigger via the 'mon' alias."""
        ns = make_need_set(description="We prefer monetary donations")
        violations = app.summarize_soft_preference_violations(
            ns, self._weekday_volunteer_row()
        )
        assert violations == []

    def test_prefers_mondays_plural_and_inflection(self, app):
        """'prefers mondays' — inflected signal word + plural day."""
        ns = make_need_set(description="The team prefers mondays")
        violations = app.summarize_soft_preference_violations(
            ns, self._weekday_volunteer_row()
        )
        assert violations == ["Does not match preferred day: Mon"]

    def test_satisfied_preference_yields_no_violation(self, app):
        ns = make_need_set(description="We prefer tuesday sessions")
        violations = app.summarize_soft_preference_violations(
            ns, self._weekday_volunteer_row()
        )
        assert violations == []

    def test_time_block_preference_word_bounded(self, app):
        ns = make_need_set(description="Ideally mornings for setup")
        row = roster_frame(
            make_volunteer("V-0001", "Eve", availability_days="Mon",
                           availability_time_blocks="Evening"),
        ).iloc[0]
        violations = app.summarize_soft_preference_violations(ns, row)
        assert violations == ["Does not match preferred time block: Morning"]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — fix 4: date-pair validation + notice visibility
# ═══════════════════════════════════════════════════════════════════════════

class TestFix4DateGuard:
    def test_inverted_dates_block_whole_roster_documented(self, app):
        """Failure-mode documentation: notification AFTER target drives the
        notice window negative and (correctly, per D2 — the check stays
        hard) blocks every volunteer on Notice Period.  The UI submit
        guard exists to stop this state from ever reaching the matcher;
        golden scenario G4 verifies it live in Phase 5.
        """
        roster = roster_frame(
            make_volunteer("V-0001", "Zero", min_notice_days=0),
            make_volunteer("V-0002", "Week", min_notice_days=7),
        )
        result = run_matching_defaults(
            app, make_need_set(), roster,
            has_specific_date=True,
            target_date_str="2026-08-01",
            notification_date_str="2026-08-05",   # after the target
        )
        assert result["matched"] == []
        blocked = result["counterfactuals"].get("Notice Period", [])
        assert {b["volunteer_id"] for b in blocked} == {"V-0001", "V-0002"}

    def test_submit_guard_present_in_input_stage(self, app):
        """Tripwire: the submit-handler guard must not be refactored away.
        (Behavior itself is browser-verified in G4 — widgets don't run
        meaningfully under bare mode.)"""
        import inspect
        src = inspect.getsource(app.render_input_stage)
        assert "notification_date > target_date" in src
        assert "notification_date < date.today()" in src

    def test_notice_window_shown_on_review_stage(self, app):
        import inspect
        src = inspect.getsource(app.render_skills_review_stage)
        assert "Notice window" in src


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — S1: native Anthropic client, call helpers, stub recommender
# ═══════════════════════════════════════════════════════════════════════════

def _canned_classifier_json(app) -> str:
    return app.ClassifierOutput(
        need_sets=[app.NeedSet(count=1, description="A helper")],
        reasoning="canned",
    ).model_dump_json()


class TestS1RequestConstruction:
    """PLAN §1a asserted on the wire — the SDK's real request serialization
    is captured by a mocked httpx transport; nothing reaches the network."""

    def test_classifier_body(self, app, monkeypatch):
        captured = []
        patch_llm(monkeypatch, app, captured, [_canned_classifier_json(app)])
        result = app.call_classifier("some request context")
        body = captured[0]
        assert body["model"] == "claude-opus-4-8"
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"]["effort"] == "medium"
        assert body["output_config"]["format"]["type"] == "json_schema"
        # Temperature is rejected outright on Opus 4.8 — assert ABSENCE.
        assert "temperature" not in body
        assert "top_p" not in body and "top_k" not in body
        assert body["system"] == app.CLASSIFIER_SYSTEM_PROMPT
        assert body["messages"] == [
            {"role": "user", "content": "some request context"}
        ]
        assert isinstance(result, app.ClassifierOutput)

    def test_classifier_tolerates_thinking_blocks(self, app, monkeypatch):
        """Adaptive thinking prepends thinking blocks; parsing must not
        assume the text block is content[0]."""
        payload = [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": _canned_classifier_json(app)},
        ]
        patch_llm(monkeypatch, app, [], [payload])
        result = app.call_classifier("ctx")
        assert result.need_sets[0].description == "A helper"

    def test_scorer_body(self, app, monkeypatch):
        captured = []
        patch_llm(monkeypatch, app, captured, [json.dumps({"selection": 4})])
        selection = app.call_likert_item("shared ctx", "profile text", "item text")
        body = captured[0]
        assert body["model"] == "claude-haiku-4-5"
        assert body["temperature"] == 0.2
        assert "thinking" not in body
        schema = body["output_config"]["format"]["schema"]
        assert schema["properties"]["selection"]["enum"] == [1, 2, 3, 4, 5]
        assert schema["additionalProperties"] is False
        assert body["system"] == "shared ctx"
        assert selection == 4

    def test_reasoning_body(self, app, monkeypatch):
        captured = []
        patch_llm(monkeypatch, app, captured, ["  A fine fit.  "])
        text = app.call_reasoning("bundle text", "tier prompt")
        body = captured[0]
        assert body["model"] == "claude-sonnet-4-6"
        assert body["temperature"] == 0.2
        assert body["max_tokens"] == 200
        assert "output_config" not in body
        assert "thinking" not in body
        assert body["system"] == "tier prompt"
        assert text == "A fine fit."

    def test_strict_schema_closes_every_object(self, app):
        """The structured-outputs grammar requires additionalProperties:
        false on ALL objects, including nested $defs."""
        schema = app._strict_schema(app.ClassifierOutput)

        violations = []

        def _walk(node, path):
            if isinstance(node, dict):
                if node.get("type") == "object" or "properties" in node:
                    if node.get("additionalProperties") is not False:
                        violations.append(path)
                for key, child in node.items():
                    _walk(child, f"{path}.{key}")
            elif isinstance(node, list):
                for i, child in enumerate(node):
                    _walk(child, f"{path}[{i}]")

        _walk(schema, "$")
        assert violations == []

    def test_client_disables_sdk_retries(self, app):
        """S3's single jittered retry must be the only retry layer."""
        import inspect
        src = inspect.getsource(app.get_anthropic_client)
        assert "max_retries=0" in src


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — S3: Likert scoring node — waves, retry, technical fallback
# ═══════════════════════════════════════════════════════════════════════════

def _match_group(idx, vids, description="Helper"):
    return {
        "need_set_index": idx,
        "need_set_description": description,
        "count_needed": 1,
        "matched_volunteer_ids": vids,
        "margins": {},
    }


def _scoring_state(app, monkeypatch, vids, **state_overrides):
    """State + roster wiring for score_volunteers_node tests."""
    roster = roster_frame(*[
        make_volunteer(vid, f"Vol {vid}") for vid in vids
    ])
    patch_loaders(monkeypatch, app, roster, assignments_frame())
    defaults = {
        "need_sets": [make_need_set()],
        "matched_volunteers": [_match_group(0, list(vids))],
    }
    defaults.update(state_overrides)
    return make_state(**defaults)


class TestS3WavePartitioning:
    def test_wave_shapes_deterministic(self, app):
        for n, shape in [(1, [1]), (4, [4]), (5, [4, 1]), (9, [4, 4, 1])]:
            units = list(range(n))
            waves = app.partition_waves(units)
            assert [len(w) for w in waves] == shape
            assert [u for w in waves for u in w] == units  # order preserved


class TestS3ScoringNode:
    def test_aggregation_raw_boxes_total(self, app, monkeypatch):
        """Known selections → raw list, box collapse, and summed score."""
        by_item = {"great fit": 5, "no friction": 4, "glad to take": 3,
                   "would recommend": 2}

        def fake_item(shared, profile, item_prompt):
            for marker, val in by_item.items():
                if marker in item_prompt:
                    return val
            raise AssertionError(f"unknown item: {item_prompt[:60]}")

        monkeypatch.setattr(app, "call_likert_item", fake_item)
        out = app.score_volunteers_node(
            _scoring_state(app, monkeypatch, ["V-0001"])
        )
        rec = out["recommendations"][0]
        assert rec["raw_selections"] == [5, 4, 3, 2]
        assert rec["boxes"] == ["T2B", "T2B", "Neutral", "B2B"]
        assert rec["total_score"] == 3 + 3 + 1 - 1  # == 6
        assert rec["tier"] == "Good Match"          # 2 ≤ 6 ≤ 8 (S4 mapping)

    def test_persistent_failure_isolated_to_its_volunteer(self, app, monkeypatch):
        """One volunteer's failing item → Technical fallback with the
        templated note; the other volunteer scores normally.  Never a
        fake Neutral."""
        monkeypatch.setattr(app.time, "sleep", lambda s: None)

        def fake_item(shared, profile, item_prompt):
            if "V-BAD" in profile and "great fit" in item_prompt:
                raise RuntimeError("persistent transport failure")
            return 5

        monkeypatch.setattr(app, "call_likert_item", fake_item)
        out = app.score_volunteers_node(
            _scoring_state(app, monkeypatch, ["V-BAD", "V-GOOD"])
        )
        recs = {r["volunteer_id"]: r for r in out["recommendations"]}
        assert recs["V-BAD"]["tier"] == "Technical Match"
        assert recs["V-BAD"]["reasoning"] == app.SCORING_UNAVAILABLE_NOTE
        assert recs["V-BAD"]["raw_selections"] is None
        assert recs["V-GOOD"]["raw_selections"] == [5, 5, 5, 5]
        assert recs["V-GOOD"]["total_score"] == 12

    def test_single_retry_recovers_transient_failure(self, app, monkeypatch):
        monkeypatch.setattr(app.time, "sleep", lambda s: None)
        attempts = {"n": 0}

        def flaky_item(shared, profile, item_prompt):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("transient")
            return 4

        monkeypatch.setattr(app, "call_likert_item", flaky_item)
        state = _scoring_state(app, monkeypatch, ["V-0001"])
        out = app.score_volunteers_node(state)
        rec = out["recommendations"][0]
        assert rec["raw_selections"] == [4, 4, 4, 4]
        assert attempts["n"] == 5  # 4 items + 1 retry

    def test_almost_matched_never_scored(self, app, monkeypatch):
        scored_profiles = []

        def fake_item(shared, profile, item_prompt):
            scored_profiles.append(profile)
            return 4

        monkeypatch.setattr(app, "call_likert_item", fake_item)
        state = _scoring_state(
            app, monkeypatch, ["V-0001"],
            almost_matched=[{
                "volunteer_id": "V-BLOCKED",
                "preferred_name": "Blocked Bob",
                "blocking_requirement": "Required Certifications",
                "blocking_column": "certifications",
            }],
        )
        out = app.score_volunteers_node(state)
        assert all("V-BLOCKED" not in p for p in scored_profiles)
        recs = {r["volunteer_id"]: r for r in out["recommendations"]}
        assert recs["V-BLOCKED"]["tier"] == "Almost Match"
        assert "Required Certifications" in recs["V-BLOCKED"]["reasoning"]

    def test_empty_matched_pool_makes_zero_calls(self, app, monkeypatch):
        calls = {"n": 0}

        def fake_item(shared, profile, item_prompt):
            calls["n"] += 1
            return 3

        monkeypatch.setattr(app, "call_likert_item", fake_item)
        state = _scoring_state(app, monkeypatch, [])
        state["matched_volunteers"] = [_match_group(0, [])]
        out = app.score_volunteers_node(state)
        assert calls["n"] == 0
        assert out["recommendations"] == []

    def test_duplicate_ids_across_groups_scored_once(self, app, monkeypatch):
        calls = {"n": 0}

        def fake_item(shared, profile, item_prompt):
            calls["n"] += 1
            return 4

        monkeypatch.setattr(app, "call_likert_item", fake_item)
        state = _scoring_state(app, monkeypatch, ["V-0001"])
        state["need_sets"] = [make_need_set(), make_need_set()]
        state["matched_volunteers"] = [
            _match_group(0, ["V-0001"]), _match_group(1, ["V-0001"]),
        ]
        out = app.score_volunteers_node(state)
        assert len(out["recommendations"]) == 1
        assert calls["n"] == 4  # one wave of four items, once


class TestS3PromptFactory:
    def test_recurring_line_present_when_set(self, app):
        state = make_state(is_recurring=True, recurring_end_date="2026-10-01")
        ctx = app.build_scorer_shared_context(state, make_need_set())
        assert "recurring need through 2026-10-01" in ctx
        assert "sustained availability" in ctx

    def test_recurring_line_absent_when_unset(self, app):
        ctx = app.build_scorer_shared_context(make_state(), make_need_set())
        assert "recurring" not in ctx

    def test_suggested_skills_labeled_context_only(self, app):
        state = make_state(unchecked_skills=["Photography/Media"])
        ctx = app.build_scorer_shared_context(state, make_need_set())
        assert "Photography/Media" in ctx
        assert "suggested but not required" in ctx

    def test_scorer_never_sees_the_word_tier(self, app):
        """I4: thresholds/caps live in code; the model must not be told."""
        state = make_state(
            soft_preferences="prefers weekends",
            unchecked_skills=["Driver"],
            is_recurring=True,
            recurring_end_date="2026-12-31",
        )
        ctx = app.build_scorer_shared_context(state, make_need_set())
        profile = app.build_volunteer_profile(
            roster_frame(make_volunteer("V-0001", "Ada")).iloc[0]
        )
        for item in app.LIKERT_ITEMS:
            prompt = app.build_item_prompt(item)
            assert "tier" not in (ctx + profile + prompt).lower()
        anchors = app.build_item_prompt(app.LIKERT_ITEMS[0])
        assert "5 = Strongly agree" in anchors
        assert "1 = Strongly disagree" in anchors


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — S4: postprocess_recommendations — thresholds, caps, sort
# ═══════════════════════════════════════════════════════════════════════════

class TestS4TierMapping:
    def test_all_nine_attainable_sums(self, app):
        """Full table: each item contributes {+3, +1, −1}; four items give
        the even sums in [−4, 12]."""
        expected = {
            -4: "Technical Match",
            -2: "Technical Match",
            0: "Technical Match",
            2: "Good Match",
            4: "Good Match",
            6: "Good Match",
            8: "Good Match",
            10: "Perfect Match",
            12: "Perfect Match",
        }
        for total, tier in expected.items():
            assert app.map_score_to_tier(total) == tier, total
        assert app.map_score_to_tier(None) == "Technical Match"

    def test_thresholds_are_module_constants(self, app):
        assert app.PERFECT_MIN == 10
        assert app.GOOD_MIN == 2


class TestS4Caps:
    def _rec(self, vid, total, name=None):
        selections = {12: [5, 5, 5, 5], 6: [5, 4, 3, 2], 0: [3, 3, 1, 1]}[total]
        return {
            "volunteer_id": vid,
            "tier": "Technical Match",
            "reasoning": "",
            "raw_selections": selections,
            "boxes": [{5: "T2B", 4: "T2B", 3: "Neutral", 2: "B2B",
                       1: "B2B"}[s] for s in selections],
            "total_score": total,
        }

    def test_cap_a_alone_demotes_perfect_to_good(self, app):
        recs = app.postprocess_recommendations(
            [self._rec("V-1", 12)],
            {"V-1": ["stated_soft_preference"]},
            {"V-1": "Ada"},
        )
        assert recs[0]["tier"] == "Good Match"
        assert recs[0]["caps_applied"] == ["stated_soft_preference"]

    def test_cap_b_alone_demotes_perfect_to_good(self, app):
        recs = app.postprocess_recommendations(
            [self._rec("V-1", 12)],
            {"V-1": ["schedule_preference"]},
            {"V-1": "Ada"},
        )
        assert recs[0]["tier"] == "Good Match"
        assert recs[0]["caps_applied"] == ["schedule_preference"]

    def test_both_caps_together_recorded(self, app):
        recs = app.postprocess_recommendations(
            [self._rec("V-1", 12)],
            {"V-1": ["stated_soft_preference", "schedule_preference"]},
            {"V-1": "Ada"},
        )
        assert recs[0]["tier"] == "Good Match"
        assert recs[0]["caps_applied"] == [
            "stated_soft_preference", "schedule_preference"
        ]

    def test_caps_cannot_promote(self, app):
        """A Technical volunteer with a violation stays Technical and
        records no applied cap."""
        recs = app.postprocess_recommendations(
            [self._rec("V-1", 0)],
            {"V-1": ["schedule_preference"]},
            {"V-1": "Ada"},
        )
        assert recs[0]["tier"] == "Technical Match"
        assert recs[0]["caps_applied"] == []

    def test_good_tier_with_cap_unchanged(self, app):
        recs = app.postprocess_recommendations(
            [self._rec("V-1", 6)],
            {"V-1": ["stated_soft_preference"]},
            {"V-1": "Ada"},
        )
        assert recs[0]["tier"] == "Good Match"
        assert recs[0]["caps_applied"] == []

    def test_sort_by_tier_rank_then_name(self, app):
        recs = app.postprocess_recommendations(
            [
                self._rec("V-TECH", 0),
                self._rec("V-ZED", 12),
                self._rec("V-ANN", 12),
                {"volunteer_id": "V-ALM", "tier": "Almost Match",
                 "reasoning": "Blocked by: X"},
            ],
            {},
            {"V-TECH": "Mid", "V-ZED": "Zed", "V-ANN": "Ann", "V-ALM": "Alm"},
        )
        assert [r["volunteer_id"] for r in recs] == [
            "V-ANN", "V-ZED", "V-TECH", "V-ALM"
        ]

    def test_almost_match_never_touched(self, app):
        entry = {"volunteer_id": "V-ALM", "tier": "Almost Match",
                 "reasoning": "Blocked by: Notice Period"}
        recs = app.postprocess_recommendations(
            [dict(entry)], {"V-ALM": ["schedule_preference"]}, {"V-ALM": "A"},
        )
        assert recs[0]["tier"] == "Almost Match"
        assert recs[0]["reasoning"] == "Blocked by: Notice Period"


class TestS4NodeIntegration:
    def test_stated_preference_violation_caps_at_good(self, app, monkeypatch):
        """Perfect-scoring volunteer who misses a stated 'prefer monday'
        lands at Good Match with the cap recorded."""
        roster = roster_frame(
            make_volunteer("V-TUE", "TueOnly", availability_days="Tue"),
        )
        patch_loaders(monkeypatch, app, roster, assignments_frame())
        monkeypatch.setattr(app, "call_likert_item", lambda *a: 5)
        state = make_state(
            soft_preferences="They prefer monday sessions",
            need_sets=[make_need_set()],
            matched_volunteers=[_match_group(0, ["V-TUE"])],
        )
        out = app.score_volunteers_node(state)
        rec = out["recommendations"][0]
        assert rec["total_score"] == 12
        assert rec["tier"] == "Good Match"
        assert rec["caps_applied"] == ["stated_soft_preference"]

    def test_missing_suggested_skills_can_still_reach_perfect(self, app, monkeypatch):
        """G5: unconfirmed suggestions are context only — never a cap."""
        roster = roster_frame(
            make_volunteer("V-0001", "Ada"),   # lacks the suggested skill
        )
        patch_loaders(monkeypatch, app, roster, assignments_frame())
        monkeypatch.setattr(app, "call_likert_item", lambda *a: 5)
        state = make_state(
            unchecked_skills=["Photography/Media"],
            need_sets=[make_need_set()],
            matched_volunteers=[_match_group(0, ["V-0001"])],
        )
        out = app.score_volunteers_node(state)
        rec = out["recommendations"][0]
        assert rec["tier"] == "Perfect Match"
        assert rec["caps_applied"] == []


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — S5: result assembly, gap notes, configuration_notes deletion
# ═══════════════════════════════════════════════════════════════════════════

class TestS5GapNotes:
    def test_covered_need_sets_produce_no_notes(self, app):
        mg = _match_group(0, ["V-0001"])
        assert app.build_gap_notes([mg], {}) == ""

    def test_underfilled_need_set_reports_need_found_and_blockers(self, app):
        mg = _match_group(0, [], description="Delivery driver")
        counterfactuals = {
            "NS0: Transportation": [{"volunteer_id": "V-1"},
                                    {"volunteer_id": "V-2"}],
            "NS0: Notice Period": [{"volunteer_id": "V-3"}],
            "NS1: Transportation": [{"volunteer_id": "V-9"}],  # other NS
        }
        notes = app.build_gap_notes([mg], counterfactuals)
        assert "Need set 'Delivery driver': need 1, found 0." in notes
        assert "Transportation blocks 2" in notes
        assert "Notice Period blocks 1" in notes
        assert "V-9" not in notes  # other need set's blockers stay out

    def test_blockers_ordered_by_count_then_name(self, app):
        mg = _match_group(0, [], description="X")
        counterfactuals = {
            "NS0: Beta": [{}, {}],
            "NS0: Alpha": [{}, {}],
            "NS0: Gamma": [{}, {}, {}],
        }
        notes = app.build_gap_notes([mg], counterfactuals)
        assert notes.index("Gamma blocks 3") < notes.index("Alpha blocks 2")
        assert notes.index("Alpha blocks 2") < notes.index("Beta blocks 2")

    def test_node_level_gap_on_fix3_style_fixture(self, app, monkeypatch):
        """The fix-3 scenario minus the car: driver slot unfillable →
        gap notes name the need set and the transportation blocker."""
        roster = roster_frame(
            make_volunteer("V-ANA", "Ana", languages="Spanish;English",
                           transportation="Public Transit"),
            make_volunteer("V-BEA", "Bea", languages="Spanish;English",
                           transportation="Public Transit"),
        )
        patch_loaders(monkeypatch, app, roster, assignments_frame())
        state = make_state(need_sets=[
            make_need_set(description="Spanish-speaking intake volunteer",
                          languages={"AND": ["Spanish"], "OR": []}),
            make_need_set(description="Delivery driver",
                          transportation_needed="Car"),
        ])
        match_out = app.match_volunteers_node(state)
        state.update(match_out)
        monkeypatch.setattr(app, "call_likert_item", lambda *a: 4)
        out = app.score_volunteers_node(state)
        assert "Need set 'Delivery driver': need 1, found 0." in out["gap_notes"]
        assert "Transportation blocks" in out["gap_notes"]

    def test_gap_notes_none_when_everything_covered(self, app, monkeypatch):
        state = _scoring_state(app, monkeypatch, ["V-0001"])
        monkeypatch.setattr(app, "call_likert_item", lambda *a: 4)
        out = app.score_volunteers_node(state)
        assert out["gap_notes"] is None


class TestS5AssemblyAndCleanup:
    def test_scored_entry_round_trips_through_sanitize(self, app):
        """Msgpack-safety across the interrupt for the new result fields."""
        entry = {
            "volunteer_id": "V-0001",
            "tier": "Good Match",
            "reasoning": "",
            "raw_selections": [5, 4, 3, 2],
            "boxes": ["T2B", "T2B", "Neutral", "B2B"],
            "total_score": 6,
            "caps_applied": ["schedule_preference"],
        }
        assert app.sanitize_for_state(entry) == entry

    def test_configuration_notes_fully_deleted(self, app):
        """D10: no prompt, no state field, no column, no banner."""
        from pathlib import Path
        src = Path(app.__file__).read_text(encoding="utf-8")
        assert "configuration_notes" not in src

    def test_almost_entry_templated_blocker_reasoning(self, app, monkeypatch):
        state = _scoring_state(
            app, monkeypatch, ["V-0001"],
            almost_matched=[{
                "volunteer_id": "V-BLOCKED",
                "preferred_name": "Blocked Bob",
                "blocking_requirement": "Notice Period",
                "blocking_column": "min_notice_days",
            }],
        )
        monkeypatch.setattr(app, "call_likert_item", lambda *a: 4)
        out = app.score_volunteers_node(state)
        recs = {r["volunteer_id"]: r for r in out["recommendations"]}
        assert recs["V-BLOCKED"]["reasoning"] == (
            "Blocked by exactly one hard requirement: Notice Period."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — S6: on-demand reasoning + dissent detection
# ═══════════════════════════════════════════════════════════════════════════

class TestS6DissentDetector:
    def test_leading_phrase_is_dissent(self, app):
        assert app.detect_dissent("On second thought, Ada is not perfect.")
        assert app.detect_dissent("on second thought — a Good fit at best.")
        assert app.detect_dissent("ON SECOND THOUGHT...")

    def test_leading_quotes_and_curly_punctuation_tolerated(self, app):
        assert app.detect_dissent('"On second thought, reconsider."')
        assert app.detect_dissent("“On second thought… not quite.”")
        assert app.detect_dissent("  'On second thought, no.'")

    def test_plain_agreement_is_not_dissent(self, app):
        assert not app.detect_dissent("Ada is a perfect fit for this role.")
        assert not app.detect_dissent("")
        assert not app.detect_dissent(None)

    def test_mid_text_mention_is_not_dissent(self, app):
        assert not app.detect_dissent(
            "Ada fits well. On second thought, she also drives."
        )


class TestS6FetchReasoning:
    def test_cache_prevents_duplicate_calls_on_rerun(self, app, monkeypatch):
        calls = {"n": 0}

        def fake_reasoning(bundle, system_prompt):
            calls["n"] += 1
            return "A fine fit."

        monkeypatch.setattr(app, "call_reasoning", fake_reasoning)
        cache = {}
        first = app.fetch_reasoning("bundle", "Good Match", cache, ("t", "V-1"))
        again = app.fetch_reasoning("bundle", "Good Match", cache, ("t", "V-1"))
        other = app.fetch_reasoning("bundle", "Good Match", cache, ("t", "V-2"))
        assert calls["n"] == 2
        assert first is again
        assert other["text"] == "A fine fit."

    def test_event_carries_dissent_flag_and_model(self, app, monkeypatch):
        monkeypatch.setattr(
            app, "call_reasoning",
            lambda bundle, prompt: "On second thought, merely good.",
        )
        event = app.fetch_reasoning("b", "Perfect Match", {}, ("t", "V-1"))
        assert event["dissent"] is True
        assert event["tier"] == "Perfect Match"   # I5: tier untouched
        assert event["model"] == app.REASONING_MODEL

    def test_tier_prompt_selected_by_tier(self, app, monkeypatch):
        seen_prompts = []

        def fake_reasoning(bundle, system_prompt):
            seen_prompts.append(system_prompt)
            return "ok"

        monkeypatch.setattr(app, "call_reasoning", fake_reasoning)
        app.fetch_reasoning("b", "Technical Match", {}, ("t", "V-1"))
        assert seen_prompts[0] == app.REASONING_TIER_PROMPTS["Technical Match"]


class TestS6BundleAndPrompts:
    def test_bundle_contains_scores_and_tier(self, app):
        vol = roster_frame(make_volunteer("V-0001", "Ada")).iloc[0]
        rec = {
            "volunteer_id": "V-0001", "tier": "Good Match", "reasoning": "",
            "raw_selections": [5, 4, 3, 2],
            "boxes": ["T2B", "T2B", "Neutral", "B2B"],
            "total_score": 6, "caps_applied": [],
        }
        bundle = app.build_reasoning_bundle(
            "Need a helper", "General help", "prefers weekends", vol, rec,
        )
        assert "Good Match" in bundle
        assert "overall_fit: 5 (T2B)" in bundle
        assert "recommendation: 2 (B2B)" in bundle
        assert "Total score: 6" in bundle
        assert "Ada" in bundle and "prefers weekends" in bundle

    def test_bundle_notes_unavailable_scoring(self, app):
        vol = roster_frame(make_volunteer("V-0001", "Ada")).iloc[0]
        rec = {"volunteer_id": "V-0001", "tier": "Technical Match",
               "reasoning": app.SCORING_UNAVAILABLE_NOTE,
               "raw_selections": None, "boxes": None, "total_score": None}
        bundle = app.build_reasoning_bundle("R", "NS", "", vol, rec)
        assert "unavailable" in bundle

    def test_tier_prompts_verbatim_anchors(self, app):
        perfect = app.REASONING_TIER_PROMPTS["Perfect Match"]
        assert perfect.startswith(
            "Review why this respondent is a perfect fit"
        )
        for tier_prompt in app.REASONING_TIER_PROMPTS.values():
            assert 'preface your reasoning with, "On second thought..."' \
                in tier_prompt
        assert set(app.REASONING_TIER_PROMPTS) == {
            "Perfect Match", "Good Match", "Technical Match",
        }  # D-H: Almost Match has no prompt and no button

    def test_ui_gates_button_away_from_almost_cards(self, app):
        import inspect
        src = inspect.getsource(app.render_results_stage)
        assert "Get reasoning" in src
        assert 'if tier == "Almost Match"' in src


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — S7: SQLite request record, schema_version 2, seed script
# ═══════════════════════════════════════════════════════════════════════════

def _record_state(app, monkeypatch, tmp_path):
    """Point the DB path at a temp file and build a fully-scored state."""
    db_path = str(tmp_path / "requests.db")
    monkeypatch.setattr(app, "REQUESTS_DB_PATH", db_path)
    state = make_state(
        need_sets=[make_need_set()],
        matched_volunteers=[_match_group(0, ["V-0001"])],
        margins={"V-0001": {"extra_skills": []}},
        recommendations=[{
            "volunteer_id": "V-0001", "tier": "Good Match", "reasoning": "",
            "raw_selections": [5, 4, 3, 2],
            "boxes": ["T2B", "T2B", "Neutral", "B2B"],
            "total_score": 6, "caps_applied": [],
        }],
        gap_notes=None,
    )
    return db_path, state


class TestS7RequestRecord:
    def test_v2_round_trip_including_raw_selections(self, app, monkeypatch, tmp_path):
        db_path, state = _record_state(app, monkeypatch, tmp_path)
        out = app.write_request_record_node(state)

        with _sqlite3.connect(db_path) as conn:
            df = pd.read_sql("SELECT * FROM requests", conn)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["schema_version"] == 2
        assert row["user_prompt"] == state["user_prompt"]
        recs = json.loads(row["recommendations_json"])
        assert recs[0]["raw_selections"] == [5, 4, 3, 2]
        assert recs[0]["boxes"] == ["T2B", "T2B", "Neutral", "B2B"]
        assert recs[0]["total_score"] == 6
        assert recs[0]["caps_applied"] == []
        # The node's return value round-trips through sanitize_for_state
        assert out["request_record"]["request_id"] == row["request_id"]

    def test_request_id_is_full_uuid(self, app, monkeypatch, tmp_path):
        _, state = _record_state(app, monkeypatch, tmp_path)
        out = app.write_request_record_node(state)
        rid = out["request_record"]["request_id"]
        assert str(_uuid.UUID(rid)) == rid    # parses, canonical form
        assert len(rid) == 36                 # full — not the old [:8]

    def test_wal_mode_active(self, app, monkeypatch, tmp_path):
        db_path, state = _record_state(app, monkeypatch, tmp_path)
        app.write_request_record_node(state)
        with _sqlite3.connect(db_path) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_concurrent_writes_all_land(self, app, monkeypatch, tmp_path):
        """Requests and reasoning events written from parallel threads."""
        from concurrent.futures import ThreadPoolExecutor
        db_path, state = _record_state(app, monkeypatch, tmp_path)
        app.init_request_db()

        def write_request(_):
            app.write_request_record_node(dict(state))

        def write_event(i):
            app.log_reasoning_event(
                f"req-{i}", f"V-{i:04d}",
                {"tier": "Good Match", "model": "claude-sonnet-4-6",
                 "text": "fine", "dissent": False},
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write_request, range(6)))
            list(pool.map(write_event, range(6)))

        with _sqlite3.connect(db_path) as conn:
            n_req = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            n_evt = conn.execute(
                "SELECT COUNT(*) FROM reasoning_events").fetchone()[0]
        assert n_req == 6
        assert n_evt == 6

    def test_dissent_stored_as_0_or_1(self, app, monkeypatch, tmp_path):
        db_path, _ = _record_state(app, monkeypatch, tmp_path)
        app.init_request_db()
        app.log_reasoning_event(
            "req-1", "V-0001",
            {"tier": "Perfect Match", "model": "claude-sonnet-4-6",
             "text": "On second thought, merely good.", "dissent": True},
        )
        app.log_reasoning_event(
            "req-1", "V-0002",
            {"tier": "Good Match", "model": "claude-sonnet-4-6",
             "text": "Solid fit.", "dissent": False},
        )
        with _sqlite3.connect(db_path) as conn:
            values = [r[0] for r in conn.execute(
                "SELECT dissent FROM reasoning_events ORDER BY id")]
        assert values == [1, 0]

    def test_reasoning_events_append_only_in_source(self, app):
        """Tripwire for the append-only hard rule: INSERT is the only
        statement ever issued against reasoning_events."""
        from pathlib import Path
        src = Path(app.__file__).read_text(encoding="utf-8")
        assert "UPDATE reasoning_events" not in src
        assert "DELETE FROM reasoning_events" not in src


class TestS7Seed:
    def _seeded_conn(self, app, monkeypatch, tmp_path):
        db_path = str(tmp_path / "requests.db")
        monkeypatch.setattr(app, "REQUESTS_DB_PATH", db_path)
        app.init_request_db()
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "seed_requests_under_test",
            Path(app.__file__).parent / "data" / "seed_requests.py",
        )
        seed_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(seed_mod)
        conn = _sqlite3.connect(db_path)
        return conn, seed_mod

    def test_seed_populates_thirty_v2_rows_and_events(self, app, monkeypatch, tmp_path):
        conn, seed = self._seeded_conn(app, monkeypatch, tmp_path)
        inserted = seed.seed_database(conn)
        conn.commit()
        assert inserted == 30
        versions = {r[0] for r in conn.execute(
            "SELECT DISTINCT schema_version FROM requests")}
        assert versions == {2}
        n_events = conn.execute(
            "SELECT COUNT(*) FROM reasoning_events").fetchone()[0]
        assert n_events >= 4
        n_dissent = conn.execute(
            "SELECT COUNT(*) FROM reasoning_events WHERE dissent = 1"
        ).fetchone()[0]
        assert n_dissent >= 1
        # Seeded recommendations parse and carry the v2 scoring fields
        rec_json = conn.execute(
            "SELECT recommendations_json FROM requests LIMIT 1").fetchone()[0]
        recs = json.loads(rec_json)
        assert "raw_selections" in recs[0] and "total_score" in recs[0]
        conn.close()

    def test_seed_idempotent(self, app, monkeypatch, tmp_path):
        conn, seed = self._seeded_conn(app, monkeypatch, tmp_path)
        first = seed.seed_database(conn)
        conn.commit()
        second = seed.seed_database(conn)
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        assert (first, second) == (30, 0)
        assert count == 30
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — permanent guards for behavior that already works.
# If one of these goes red, the change is wrong — not the test.
# ═══════════════════════════════════════════════════════════════════════════

class TestGuardCommittedHours:
    def test_messy_frame_iso_week_exactly_five(self, app):
        """The canonical messy 5-row frame: exactly 5.0 committed hours in
        the ISO week of 2026-07-23 (Mon 07-20 … Sun 07-26).

        Covers status normalization (Confirmed/COMPLETE variants count;
        no-show variants don't), unparseable dates skipped gracefully,
        and the prior-week boundary.
        """
        raw = assignments_frame([
            {"volunteer_id": "V-0001", "status": "Confirmed",
             "start_date": "2026-07-23", "hours_required": 2.0},
            {"volunteer_id": "V-0001", "status": "COMPLETE",
             "start_date": "2026-07-24", "hours_required": 3.0},
            {"volunteer_id": "V-0001", "status": "no-show",
             "start_date": "2026-07-23", "hours_required": 4.0},
            {"volunteer_id": "V-0001", "status": "Confirmed",
             "start_date": "not-a-date", "hours_required": 6.0},
            {"volunteer_id": "V-0001", "status": "confirmed",
             "start_date": "2026-07-14", "hours_required": 8.0},
        ])
        # Mirror load_assignments' status normalization on the messy input.
        raw["status"] = raw["status"].apply(app.normalize_assignment_status)
        assert app.get_committed_hours("V-0001", "2026-07-23", raw) == 5.0

    def test_no_target_date_is_zero(self, app):
        assert app.get_committed_hours("V-0001", None, assignments_frame()) == 0.0

    def test_unparseable_target_date_is_zero(self, app):
        assert app.get_committed_hours(
            "V-0001", "someday", assignments_frame()
        ) == 0.0


class TestGuardDegenerateBranches:
    def test_na_and_values_do_not_block(self, app):
        """Runtime-side degeneracy guard: NA/empty AND values are ignored
        during evaluation, independent of normalization."""
        assert app.evaluate_flexible_requirement(
            {"AND": ["", "NA"], "OR": []}, set()
        )

    def test_empty_values_inside_or_branch_ignored(self, app):
        assert app.evaluate_flexible_requirement(
            {"AND": [], "OR": [["", "Mon"]]}, {"Mon"}
        )

    def test_vacuous_requirement_is_satisfied(self, app):
        assert app.evaluate_flexible_requirement({"AND": [], "OR": []}, set())


class TestGuardPerNeedSetScoping:
    def test_each_need_set_enforces_only_its_own_skills(self, app, monkeypatch):
        """Globally-confirmed skills apply per need set only where the
        classifier assigned them — 'one driver + one intake volunteer'
        must not demand both skills of every slot."""
        roster = roster_frame(
            make_volunteer(
                "V-DANA", "Dana", skills="Driver",
                certifications="Driver Authorization - Approved",
            ),
            make_volunteer(
                "V-IRIS", "Iris", skills="Intake/Translation",
                certifications="Food Safety - Basic",
            ),
        )
        patch_loaders(monkeypatch, app, roster, assignments_frame())
        state = make_state(
            confirmed_skills=["Driver", "Intake/Translation"],
            need_sets=[
                make_need_set(description="Delivery driver",
                              applicable_skills=["Driver"]),
                make_need_set(description="Intake volunteer",
                              applicable_skills=["Intake/Translation"]),
            ],
        )
        out = app.match_volunteers_node(state)
        driver_ns, intake_ns = out["matched_volunteers"]
        assert driver_ns["matched_volunteer_ids"] == ["V-DANA"]
        assert intake_ns["matched_volunteer_ids"] == ["V-IRIS"]


class TestGuardSanitizeForState:
    def test_numpy_and_pandas_scalars_round_trip(self, app):
        import numpy as np
        raw = {
            "an_int": np.int64(7),
            "a_float": np.float64(2.5),
            "a_bool": np.bool_(True),
            "a_ts": pd.Timestamp("2026-07-23T10:00:00"),
            "nested": [{"n": np.int64(1)}, (np.float64(0.5),)],
        }
        clean = app.sanitize_for_state(raw)
        assert clean["an_int"] == 7 and type(clean["an_int"]) is int
        assert clean["a_float"] == 2.5 and type(clean["a_float"]) is float
        assert clean["a_bool"] is True
        assert clean["a_ts"] == "2026-07-23T10:00:00"
        assert clean["nested"][0]["n"] == 1
        assert clean["nested"][1] == [0.5]

    def test_plain_data_idempotent(self, app):
        plain = {"a": 1, "b": [1.5, "x", True, None], "c": {"d": "e"}}
        assert app.sanitize_for_state(plain) == plain
        assert app.sanitize_for_state(app.sanitize_for_state(plain)) == plain


class TestS1MigrationComplete:
    def test_no_langchain_or_old_recommender_remnants(self, app):
        """PLAN Phase 2 exit grep, enforced early: the old single-call
        recommender is deleted, not migrated."""
        from pathlib import Path
        src = Path(app.__file__).read_text(encoding="utf-8")
        for pattern in ("RECOMMENDER_SYSTEM_PROMPT", "RecommenderOutput",
                        "VolunteerRecommendation", "ChatOpenAI", "langchain"):
            assert pattern not in src, f"stale reference: {pattern}"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — S2: LIKERT_ITEMS + SCORE_MAP source-of-truth constants
# ═══════════════════════════════════════════════════════════════════════════

class TestS2LikertConstants:
    def _source_blocks(self):
        """Parse the committed verbatim source into per-item prompt texts.

        Blocks split on blank lines; anchor-table rows (the Selection /
        Structured Output / Score columns) are stripped; remaining lines
        are whitespace-normalized and joined.
        """
        from pathlib import Path
        raw = Path(__file__).parent.joinpath(
            "data", "plaintext_ranking_prompts.txt"
        ).read_text(encoding="utf-8")
        table_markers = ("Selection:", "Strongly agree", "Somewhat agree",
                         "Neutral", "Somewhat disagree", "Strongly disagree")
        blocks = []
        for block in raw.split("\n\n"):
            lines = [
                ln.strip() for ln in block.splitlines()
                if ln.strip() and not ln.strip().startswith(table_markers)
            ]
            if lines:
                blocks.append("\n".join(lines))
        return blocks

    def test_item_texts_match_source_file_verbatim(self, app):
        source_texts = self._source_blocks()
        app_texts = [item["text"] for item in app.LIKERT_ITEMS]
        assert app_texts == source_texts

    def test_four_items_in_spec_order(self, app):
        assert [i["key"] for i in app.LIKERT_ITEMS] == [
            "overall_fit", "schedule_friction", "willingness", "recommendation",
        ]

    def test_score_map_totality_and_values(self, app):
        """Total over 1–5; T2B +3, Neutral +1, B2B −1 per the source table."""
        assert set(app.SCORE_MAP.keys()) == {1, 2, 3, 4, 5}
        assert app.SCORE_MAP == {5: 3, 4: 3, 3: 1, 2: -1, 1: -1}

    def test_anchor_labels_match_source(self, app):
        assert app.LIKERT_ANCHORS == (
            ("Strongly agree", 5),
            ("Somewhat agree", 4),
            ("Neutral", 3),
            ("Somewhat disagree", 2),
            ("Strongly disagree", 1),
        )


# ═══════════════════════════════════════════════════════════════════════════
# §12 — aesthetics pass (activated 2026-07-22): theme, cards, chips, dissent
# ═══════════════════════════════════════════════════════════════════════════

class TestS12Theme:
    def test_config_toml_exists_and_pins_the_brand_theme(self, app):
        """Pinned light + trust blue: the public demo must look the same for
        every viewer, so the theme lives in committed config, not viewer
        preference."""
        import tomllib
        from pathlib import Path
        cfg_path = Path(app.__file__).parent / ".streamlit" / "config.toml"
        assert cfg_path.exists()
        cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        theme = cfg["theme"]
        assert theme["base"] == "light"
        assert theme["primaryColor"] == "#2563EB"
        assert "sidebar" in theme


class TestS12Css:
    def test_brand_css_covers_all_four_tier_card_slugs(self, app):
        for slug in ("perfect", "good", "technical", "almost"):
            assert f'st-key-card-{slug}-' in app._BRAND_CSS

    def test_css_injected_once_in_main(self, app):
        import inspect
        assert "inject_brand_css()" in inspect.getsource(app.main)


class TestS12Cards:
    def test_results_stage_uses_real_bordered_cards_not_raw_html(self, app):
        """The old tier-color <div> strip never wrapped Streamlit children;
        cards are now keyed bordered containers styled via _BRAND_CSS."""
        import inspect
        src = inspect.getsource(app.render_results_stage)
        assert "unsafe_allow_html" not in src
        assert "st.container(border=True" in src
        assert 'key=f"card-' in src

    def test_tier_style_covers_all_four_tiers(self, app):
        import inspect
        src = inspect.getsource(app.render_results_stage)
        for slug in ("perfect", "good", "technical", "almost"):
            assert f'"{slug}"' in src
