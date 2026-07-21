"""Seed the demo request history — invoked by app.main() on first run.

Builds ~30 plausible schema_version-2 request rows plus a handful of
seeded reasoning events (at least one dissent) so the public demo opens
with a lived-in request history and the dissent-rate QA metric has data
from day one.  Deterministic content: no randomness, timestamps anchored
to a fixed base date.  The caller creates the schema (app.init_request_db)
and passes an open connection; seeding is idempotent — a non-empty
requests table is left untouched.
"""

import json
import uuid
from datetime import date, timedelta

SCHEMA_VERSION = 2
_BASE_DAY = date(2026, 6, 1)

# (prompt, skill, languages, tier mix) templates cycled across the rows.
_SCENARIOS = [
    ("Need 2 volunteers for Saturday morning pantry — one should speak "
     "Spanish for intake.", "Pantry Operations", ["Spanish"]),
    ("Looking for a math tutor for our after-school program on Thursday "
     "afternoons.", "Tutoring - Math", []),
    ("One delivery driver for weekend food distribution, must have a car.",
     "Driver", []),
    ("Volunteer to help with data entry and reporting, weekday mornings "
     "preferred.", "Data Entry", []),
    ("Two event-support volunteers for the community fair, Sunday midday.",
     "Event Support", []),
    ("ESL conversation partner needed for Tuesday evenings, ideally "
     "patient with beginners.", "ESL Support", []),
]

# (selections, expected tier) pairs consistent with the S4 thresholds.
_SCORE_SHAPES = [
    ([5, 5, 5, 4], 12, "Perfect Match"),
    ([5, 4, 3, 4], 10, "Perfect Match"),
    ([5, 4, 3, 2], 6, "Good Match"),
    ([4, 3, 3, 3], 6, "Good Match"),
    ([3, 3, 2, 2], 0, "Technical Match"),
]

_BOX = {5: "T2B", 4: "T2B", 3: "Neutral", 2: "B2B", 1: "B2B"}


def _make_row(i: int) -> dict:
    prompt, skill, langs = _SCENARIOS[i % len(_SCENARIOS)]
    selections, total, tier = _SCORE_SHAPES[i % len(_SCORE_SHAPES)]
    day = _BASE_DAY + timedelta(days=i)
    vid_a = f"V-{1000 + (i * 2) % 40:04d}"
    vid_b = f"V-{1001 + (i * 2) % 40:04d}"

    recommendations = [
        {
            "volunteer_id": vid_a,
            "tier": tier,
            "reasoning": "",
            "raw_selections": selections,
            "boxes": [_BOX[s] for s in selections],
            "total_score": total,
            "caps_applied": [],
        },
        {
            "volunteer_id": vid_b,
            "tier": "Technical Match",
            "reasoning": "",
            "raw_selections": [3, 3, 3, 3],
            "boxes": ["Neutral"] * 4,
            "total_score": 4,
            "caps_applied": [],
        },
    ]
    need_set = {
        "count": 1,
        "description": prompt,
        "applicable_skills": [skill],
        "availability_days": {"AND": [], "OR": []},
        "availability_time_blocks": [],
        "languages": {"AND": langs, "OR": []},
        "min_hours": None,
        "location_area": None,
        "transportation_needed": None,
    }
    return {
        "request_id": str(uuid.uuid5(uuid.NAMESPACE_URL,
                                     f"nca-seed-request-{i}")),
        "schema_version": SCHEMA_VERSION,
        "timestamp": f"{day.isoformat()}T09:{i % 60:02d}:00",
        "user_prompt": prompt,
        "soft_preferences": "" if i % 3 else "Prefers weekend availability",
        "unchecked_skills": "[]",
        "request_source": "seed",
        "need_sets_json": json.dumps([need_set]),
        "confirmed_skills_json": json.dumps([skill]),
        "extracted_skills_json": json.dumps([skill]),
        "form_certs_json": "[]",
        "form_languages_json": json.dumps(langs),
        "has_specific_date": int(i % 2 == 0),
        "target_date": (day + timedelta(days=7)).isoformat() if i % 2 == 0 else "",
        "notification_date": day.isoformat(),
        "is_recurring": int(i % 5 == 0),
        "matched_volunteers_json": json.dumps([{
            "need_set_index": 0,
            "need_set_description": prompt,
            "count_needed": 1,
            "matched_volunteer_ids": [vid_a, vid_b],
            "margins": {},
        }]),
        "margins_json": "{}",
        "counterfactuals_json": "{}",
        "almost_matched_json": "[]",
        "recommendations_json": json.dumps(recommendations),
        "gap_notes": "",
        "resulting_assignment_ids": "[]",
    }


def _make_events(rows: list) -> list:
    """A handful of reasoning events over the seeded rows — one dissent."""
    texts = [
        ("Perfect Match",
         "Their preferred roles and weekend availability line up directly "
         "with this request, making them the natural first call."),
        ("Good Match",
         "Strong skills overlap and easy scheduling keep them near the "
         "top, though the stated language preference is unmet."),
        ("Technical Match",
         "They clear every hard requirement, but the role sits outside "
         "their stated preferences."),
        ("Perfect Match",
         "On second thought, the tight Thursday window conflicts with "
         "their notice preference, so this looks closer to a good fit."),
        ("Good Match",
         "Availability and certifications align well; only the suggested "
         "photography skill is missing."),
        ("Technical Match",
         "Qualified on paper, though their notes suggest they prefer "
         "warehouse work over front-line intake."),
    ]
    events = []
    for j, (tier, text) in enumerate(texts):
        row = rows[j * 4 % len(rows)]
        recs = json.loads(row["recommendations_json"])
        events.append((
            row["request_id"],
            recs[0]["volunteer_id"],
            tier,
            "claude-sonnet-4-6",
            text,
            1 if text.startswith("On second thought") else 0,
            f"{row['timestamp'][:10]}T15:{j:02d}:00",
        ))
    return events


def seed_database(conn) -> int:
    """Insert the seed rows; no-op when the table already has data."""
    existing = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    if existing:
        return 0

    rows = [_make_row(i) for i in range(30)]
    columns = list(rows[0].keys())
    conn.executemany(
        f"INSERT INTO requests ({', '.join(columns)}) "
        f"VALUES ({', '.join(':' + c for c in columns)})",
        rows,
    )
    conn.executemany(
        "INSERT INTO reasoning_events "
        "(request_id, volunteer_id, tier, model, text, dissent, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        _make_events(rows),
    )
    return len(rows)
