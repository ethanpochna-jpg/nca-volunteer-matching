"""Regression suite — grows one test group per PLAN item.

Layout (kept in PLAN order as fixes land):
  Phase 0  — harness smoke
  Phase 1  — fixes 2, 10, 12, 1, 3, 5, 6, 4
  Phase 2  — S1–S7
  Phase 3  — permanent guards
"""

from tests.fixtures import (  # noqa: F401  (builders used as fixes land)
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

def _fake_classifier(app, monkeypatch, canned_output):
    """Stub ChatOpenAI so classify_needs_node runs offline.

    Exercises the REAL post-LLM sanitization code; only the network call
    is replaced.  Dies with the langchain call sites in S1.
    """
    class _Fake:
        def __init__(self, *args, **kwargs):
            pass

        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            return canned_output

    monkeypatch.setattr(app, "ChatOpenAI", _Fake)


class TestFix10SkillsCanonicalization:
    def test_lowercase_skill_survives_into_extracted_skills(self, app, monkeypatch):
        """'tutoring - math' from the classifier must reach the review UI."""
        canned = app.ClassifierOutput(
            need_sets=[app.NeedSet(
                count=1,
                description="Math tutor",
                applicable_skills=["tutoring - math", "Tutoring - Math"],
            )],
            reasoning="canned",
        )
        _fake_classifier(app, monkeypatch, canned)
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
