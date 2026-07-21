"""Request-record persistence — WAL-mode SQLite, schema_version 2.

Phase 4 verbatim move out of app.py (SECTION 7D and the DB path
constant).  Zero behavior edits.
"""

import sqlite3
from datetime import datetime
from typing import Optional

# requests.db is generated (gitignored), WAL-mode SQLite, seeded on first
# run by data/seed_requests.py.
REQUESTS_DB_PATH = "requests.db"

SCHEMA_VERSION = 2

_REQUESTS_COLUMNS = [
    "request_id", "schema_version", "timestamp", "user_prompt",
    "soft_preferences", "unchecked_skills", "request_source",
    "need_sets_json", "confirmed_skills_json", "extracted_skills_json",
    "form_certs_json", "form_languages_json", "has_specific_date",
    "target_date", "notification_date", "is_recurring",
    "matched_volunteers_json", "margins_json", "counterfactuals_json",
    "almost_matched_json", "recommendations_json", "gap_notes",
    "resulting_assignment_ids",
]


def db_connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Short-lived WAL connection; every writer opens its own."""
    conn = sqlite3.connect(db_path or REQUESTS_DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_request_db(db_path: Optional[str] = None) -> None:
    """Create both tables if absent.  Idempotent."""
    with db_connect(db_path) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL DEFAULT {SCHEMA_VERSION},
                timestamp TEXT NOT NULL,
                user_prompt TEXT NOT NULL,
                soft_preferences TEXT,
                unchecked_skills TEXT,
                request_source TEXT,
                need_sets_json TEXT,
                confirmed_skills_json TEXT,
                extracted_skills_json TEXT,
                form_certs_json TEXT,
                form_languages_json TEXT,
                has_specific_date INTEGER,
                target_date TEXT,
                notification_date TEXT,
                is_recurring INTEGER,
                matched_volunteers_json TEXT,
                margins_json TEXT,
                counterfactuals_json TEXT,
                almost_matched_json TEXT,
                recommendations_json TEXT,
                gap_notes TEXT,
                resulting_assignment_ids TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                volunteer_id TEXT NOT NULL,
                tier TEXT NOT NULL,
                model TEXT NOT NULL,
                text TEXT NOT NULL,
                dissent INTEGER NOT NULL CHECK (dissent IN (0, 1)),
                created_at TEXT NOT NULL
            )
        """)


def insert_request_record(record: dict, db_path: Optional[str] = None) -> None:
    """One request row, one transaction."""
    placeholders = ", ".join(f":{col}" for col in _REQUESTS_COLUMNS)
    with db_connect(db_path) as conn:
        conn.execute(
            f"INSERT INTO requests ({', '.join(_REQUESTS_COLUMNS)}) "
            f"VALUES ({placeholders})",
            record,
        )


def log_reasoning_event(request_id: str, volunteer_id: str, event: dict,
                        db_path: Optional[str] = None) -> None:
    """Append one reasoning event — one row per button fetch.

    INSERT only; this table is never UPDATEd and never DELETEd from.
    """
    with db_connect(db_path) as conn:
        conn.execute(
            "INSERT INTO reasoning_events "
            "(request_id, volunteer_id, tier, model, text, dissent, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                request_id,
                volunteer_id,
                event["tier"],
                event["model"],
                event["text"],
                1 if event.get("dissent") else 0,
                datetime.now().isoformat(),
            ),
        )


