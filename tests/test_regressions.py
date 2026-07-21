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
