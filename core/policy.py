"""Organizational policy — roster vocabulary and mandatory-cert rules.

Phase 4 verbatim move out of app.py (SECTION 2 vocabulary + cert rules +
infer_mandatory_certs).  Zero behavior edits.  These constants are the
source of truth end-to-end: the classifier prompt, canonicalization,
matcher, and scorer context all reference them — never let them drift.
"""


# ── Roster vocabulary ─────────────────────────────────────────────────────

VALID_SKILLS = [
    "Adult Learning", "Analytics/Reporting", "Community Outreach",
    "Crafts/Activities", "Customer Service", "Data Entry", "Driver",
    "ESL Support", "Environmental Cleanup", "Event Support",
    "Forklift (experienced)", "Intake/Translation", "Inventory/Sorting",
    "Pantry Operations", "Photography/Media", "Program Support",
    "Scheduling/Coordination", "Team Lead", "Tool Safety",
    "Training/Onboarding", "Tutoring - Math", "Tutoring - Reading",
    "Tutoring - SAT Prep", "Tutoring - Science", "Volunteer Training",
    "Warehouse/Lifting", "Youth Mentoring",
]

# Only "cleared"/"approved"/"completed" certifications count as held.
# "Pending" variants mean the volunteer is NOT yet certified and must be
# treated as if the cert is absent for matching purposes.
VALID_CERTS_CLEARABLE = [
    "Background Check - Cleared",
    "Child Safety Training - Completed",
    "Food Safety - Basic",
    "Driver Authorization - Approved",
    "Tool Safety Briefing - Completed",
    "Trainer - Approved",
    "Waiver - Signed",
]

VALID_LANGUAGES = [
    "Arabic", "Bengali", "English", "French", "Japanese", "Korean",
    "Russian", "Spanish", "Swahili", "Tamil", "Twi", "Urdu", "Vietnamese",
]

VALID_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

VALID_TIME_BLOCKS = ["Morning", "Midday", "Afternoon", "Evening"]

VALID_AREAS = [
    "Northbridge - Downtown", "Northbridge - Eastview",
    "Northbridge - Maplewood", "Northbridge - Riverbend",
    "Northbridge - South Market", "Northbridge - Westside",
]

# ── Value aliases (from app_5) ───────────────────────────────────────────
# Maps natural-language variants and common misspellings to canonical roster
# vocabulary.  The canonicalize_value() function uses these to normalize
# LLM outputs and user inputs before matching.
VALUE_ALIASES = {
    "languages": {
        "spanish-speaking": "Spanish", "spanish speaking": "Spanish",
        "speak spanish": "Spanish",
        "arabic-speaking": "Arabic", "arabic speaking": "Arabic",
        "bengali-speaking": "Bengali", "bengali speaking": "Bengali",
        "english-speaking": "English", "english speaking": "English",
        "french-speaking": "French", "french speaking": "French",
        "japanese-speaking": "Japanese", "japanese speaking": "Japanese",
        "korean-speaking": "Korean", "korean speaking": "Korean",
        "russian-speaking": "Russian", "russian speaking": "Russian",
        "swahili-speaking": "Swahili", "swahili speaking": "Swahili",
        "tamil-speaking": "Tamil", "tamil speaking": "Tamil",
        "twi-speaking": "Twi", "twi speaking": "Twi",
        "urdu-speaking": "Urdu", "urdu speaking": "Urdu",
        "vietnamese-speaking": "Vietnamese", "vietnamese speaking": "Vietnamese",
    },
    "days": {
        "monday": "Mon", "mon": "Mon",
        "tuesday": "Tue", "tue": "Tue", "tues": "Tue",
        "wednesday": "Wed", "wed": "Wed",
        "thursday": "Thu", "thu": "Thu", "thurs": "Thu",
        "friday": "Fri", "fri": "Fri",
        "saturday": "Sat", "sat": "Sat",
        "sunday": "Sun", "sun": "Sun",
    },
    "time_blocks": {
        "am": "Morning", "morning": "Morning",
        "midday": "Midday", "mid-day": "Midday", "noon": "Midday",
        "lunch": "Midday",
        "afternoon": "Afternoon", "pm": "Afternoon",
        "evening": "Evening", "night": "Evening",
    },
    "transportation": {
        "car": "Car", "vehicle": "Car", "own car": "Car",
        "public transit": "Public Transit", "transit": "Public Transit",
        "bus": "Public Transit", "train": "Public Transit",
        "bike": "Bike", "bicycle": "Bike",
        "walk": "Walk", "walking": "Walk",
        "any": "Any",
    },
}

# ── Mandatory certification rules (Coordination Manual, Section 3) ────────
# These are ORGANIZATIONAL policy, not per-request.  The matcher enforces
# them automatically when the confirmed skills trigger a category.
MANDATORY_CERT_RULES = {
    "youth_facing": [                           # Any tutoring / youth role
        "Background Check - Cleared",
        "Child Safety Training - Completed",
    ],
    "food_handling": [                          # Pantry / food contact roles
        "Food Safety - Basic",
    ],
    "driving": [                                # Delivery / transport roles
        "Driver Authorization - Approved",
    ],
}

# Which skills trigger which mandatory-cert category.
YOUTH_FACING_SKILLS = {
    "Tutoring - Math", "Tutoring - Reading", "Tutoring - Science",
    "Tutoring - SAT Prep", "ESL Support", "Youth Mentoring",
    "Crafts/Activities",
}
FOOD_HANDLING_SKILLS = {
    "Pantry Operations", "Inventory/Sorting", "Intake/Translation",
}
DRIVING_SKILLS = {"Driver"}


def infer_mandatory_certs(confirmed_skills: list) -> list:
    """Determine which certifications are ORGANIZATIONALLY REQUIRED.

    These are non-negotiable policy rules from the Coordination Manual:
    - Youth-facing skills → Background Check + Child Safety Training
    - Food handling skills → Food Safety
    - Driving skills → Driver Authorization
    """
    mandatory = set()
    skill_set = set(confirmed_skills)

    if skill_set & YOUTH_FACING_SKILLS:
        mandatory.update(MANDATORY_CERT_RULES["youth_facing"])
    if skill_set & FOOD_HANDLING_SKILLS:
        mandatory.update(MANDATORY_CERT_RULES["food_handling"])
    if skill_set & DRIVING_SKILLS:
        mandatory.update(MANDATORY_CERT_RULES["driving"])

    return list(mandatory)
