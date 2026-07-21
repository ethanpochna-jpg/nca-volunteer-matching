"""Data loading, canonicalization, and the deterministic matcher.

Phase 4 verbatim move out of app.py (SECTIONs 3, 3B, and 7).  Zero
behavior edits; cross-module references are import-qualified only.
"""

import re
from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st

from core import policy, schemas


# File paths — data files live under data/ at the repo root; the app and
# the test suite both run from the repo root, so relative paths suffice.
ROSTER_PATH = "data/northbridge_volunteer_roster.csv"
ASSIGNMENTS_PATH = "data/northbridge_volunteer_assignments.xlsx"

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
    alias_map = policy.VALUE_ALIASES.get(domain, {})
    raw_lower = raw.casefold()
    if raw_lower in alias_map:
        return alias_map[raw_lower]

    # Fall back to case-insensitive match against valid values
    valid_values = {
        "languages": policy.VALID_LANGUAGES,
        "days": policy.VALID_DAYS,
        "time_blocks": policy.VALID_TIME_BLOCKS,
        "skills": policy.VALID_SKILLS,
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

    # Fix 6: word-boundary regexes, one per CANONICAL value.  Substring
    # matching double-fired ("prefer monday" hit both the "mon" and
    # "monday" aliases) and false-positived ("prefer monetary donations"
    # hit "mon").  These violations feed the Phase 2 tier caps, so
    # precision matters.
    signal = r"(?:prefer(?:red|s)?|ideally)"
    if not re.search(rf"\b{signal}\b", desc_l):
        return []

    violations = []
    vol_days = parse_semicolon(vol_row.get("availability_days", ""), domain="days")
    vol_blocks = parse_semicolon(
        vol_row.get("availability_time_blocks", ""), domain="time_blocks"
    )

    aliases_by_day: dict[str, list[str]] = {}
    for raw, norm in policy.VALUE_ALIASES.get("days", {}).items():
        aliases_by_day.setdefault(norm, []).append(raw)

    for norm in policy.VALID_DAYS:                     # canonical order, one hit max
        aliases = sorted(aliases_by_day.get(norm, []), key=len, reverse=True)
        if not aliases:
            continue
        pattern = rf"\b{signal}\s+(?:{'|'.join(map(re.escape, aliases))})s?\b"
        if re.search(pattern, desc_l) and norm not in vol_days:
            violations.append(f"Does not match preferred day: {norm}")

    for block in policy.VALID_TIME_BLOCKS:
        pattern = rf"\b{signal}\s+{block.lower()}s?\b"
        if re.search(pattern, desc_l) and block not in vol_blocks:
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

    # Certs: explicit form selections + policy-mandated certs.
    # Fix 1 — safety fails closed: the policy basis is the need set's WORK
    # TYPE (applicable_skills ∪ confirmed_skills), never checkbox diligence
    # alone.  With zero confirmed skills a youth-tutoring request must still
    # require Background Check + Child Safety.  Accepted trade-off: the
    # classifier's inclusive suggestions can over-trigger certs; the review
    # step displays them as the mitigation.
    policy_basis = set(confirmed_skills) | set(need_set.get("applicable_skills", []))
    mandatory_certs = policy.infer_mandatory_certs(sorted(policy_basis))
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

    return schemas.sanitize_for_state({
        "matched": matched,
        "margins": margins,
        "counterfactuals": counterfactuals,
        "almost_matched": almost_matched_list,
    })


