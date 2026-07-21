"""Fixture builders shared across the regression suite.

Builders produce DataFrames in POST-load shape — i.e. what load_roster /
load_assignments return after their defensive typing — because tests patch
the loaders and hand these frames straight to the matcher.  Numeric columns
are real numerics, volunteer_id is stripped, assignments carry
start_date_parsed.  run_matching also accepts these frames directly as
parameters, which most matcher-level tests use in preference to patching.
"""

import pandas as pd

# Column order mirrors REQUIRED_ROSTER_COLUMNS plus the optional columns the
# matcher and UI read via .get(): keep in sync with app.py SECTION 3.
_VOLUNTEER_DEFAULTS = {
    "volunteer_id": "V-0000",
    "preferred_name": "Someone",
    "pronouns": "",
    "skills": "",
    "certifications": "",
    "languages": "English",
    "availability_days": "Mon;Tue;Wed;Thu;Fri;Sat;Sun",
    "availability_time_blocks": "Morning;Midday;Afternoon;Evening",
    "availability_notes": "",
    "max_hours_per_week": 40.0,
    "min_notice_days": 0.0,
    "status": "Active",
    "preferred_roles": "",
    "home_area": "Northbridge - Downtown",
    "transportation": "Public Transit",
    "notes": "",
}


def make_volunteer(volunteer_id: str, preferred_name: str, **overrides) -> dict:
    """One roster row as a dict; override any column by keyword."""
    row = dict(_VOLUNTEER_DEFAULTS)
    row["volunteer_id"] = volunteer_id
    row["preferred_name"] = preferred_name
    row.update(overrides)
    return row


def roster_frame(*volunteers: dict) -> pd.DataFrame:
    """Roster DataFrame in post-load shape (numerics coerced, ids stripped)."""
    df = pd.DataFrame(list(volunteers))
    df["max_hours_per_week"] = pd.to_numeric(df["max_hours_per_week"], errors="coerce").fillna(40)
    df["min_notice_days"] = pd.to_numeric(df["min_notice_days"], errors="coerce").fillna(0)
    df["volunteer_id"] = df["volunteer_id"].astype(str).str.strip()
    return df


def assignments_frame(rows: list[dict] | None = None) -> pd.DataFrame:
    """Assignments DataFrame in post-load shape.

    Each row dict needs volunteer_id / status / start_date / hours_required;
    status should already be canonical unless the test is exercising
    normalize_assignment_status explicitly.
    """
    if not rows:
        df = pd.DataFrame(columns=["volunteer_id", "status", "start_date", "hours_required"])
    else:
        df = pd.DataFrame(rows)
    df["volunteer_id"] = df["volunteer_id"].astype(str).str.strip()
    df["hours_required"] = pd.to_numeric(df["hours_required"], errors="coerce").fillna(0.0)
    df["start_date_parsed"] = pd.to_datetime(df["start_date"], errors="coerce")
    return df


def make_need_set(**overrides) -> dict:
    """A NeedSet dict in sanitized (post-classifier) shape."""
    ns = {
        "count": 1,
        "description": "A volunteer",
        "applicable_skills": [],
        "availability_days": {"AND": [], "OR": []},
        "availability_time_blocks": [],
        "languages": {"AND": [], "OR": []},
        "min_hours": None,
        "location_area": None,
        "transportation_needed": None,
    }
    ns.update(overrides)
    return ns


def make_state(**overrides) -> dict:
    """Full GraphState dict for node-level tests (match_volunteers_node etc.)."""
    state = {
        "user_prompt": "Need a volunteer",
        "form_certs": [],
        "form_languages": [],
        "has_specific_date": False,
        "target_date": None,
        "notification_date": "2026-07-21",
        "is_recurring": False,
        "recurring_end_date": None,
        "need_sets": [make_need_set()],
        "extracted_skills": [],
        "classifier_reasoning": "",
        "soft_preferences": "",
        "confirmed_skills": [],
        "unchecked_skills": [],
        "matched_volunteers": [],
        "margins": {},
        "counterfactuals": {},
        "almost_matched": [],
        "volunteer_histories": {},
        "recommendations": [],
        "gap_notes": None,
        "request_record": {},
    }
    state.update(overrides)
    return state


def run_matching_defaults(app, need_set, roster_df, assignments_df=None, **overrides):
    """Call app.run_matching with neutral defaults; override any kwarg.

    Keeps individual tests focused on the one requirement they exercise.
    """
    kwargs = {
        "need_set": need_set,
        "confirmed_skills": [],
        "form_certs": [],
        "form_languages": [],
        "has_specific_date": False,
        "target_date_str": None,
        "notification_date_str": "2026-07-21",
        "roster_df": roster_df,
        "assignments_df": assignments_frame() if assignments_df is None else assignments_df,
    }
    kwargs.update(overrides)
    return app.run_matching(**kwargs)
