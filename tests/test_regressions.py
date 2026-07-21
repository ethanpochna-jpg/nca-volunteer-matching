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
