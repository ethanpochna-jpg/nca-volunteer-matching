# NCA Volunteer Matching

**Live demo:** https://nca-volunteer-matching.streamlit.app

Matching Agent of a two-agent volunteer-coordination system for Northbridge
Community Alliance. Streamlit + LangGraph, all-Anthropic model stack.

A program manager describes a volunteer need in plain language. Claude Opus
extracts structured need sets (skills, days, languages, counts), a human
reviews and confirms the extraction, and a deterministic nine-check matcher
filters the roster — safety-critical certifications are inferred from the
type of work and enforced in code, never left to checkbox diligence. Matched
volunteers are then scored by Claude Haiku on four Likert items; tier
assignment (Perfect / Good / Technical / Almost Match) is pure code —
thresholds, caps, and ordering never live in a prompt. Each recommendation
card offers on-demand reasoning from Claude Sonnet, logged to an append-only
event table with dissent detection ("On second thought…" never mutates a
tier — it is flagged and displayed verbatim).

## Demo dataset

The roster (`data/northbridge_volunteer_roster.csv`) and assignment history
(`data/northbridge_volunteer_assignments.xlsx`) are synthetic course data for
a fictional organization. `requests.db` is generated SQLite, seeded with 30
deterministic demo requests on first boot; on Streamlit Community Cloud it
lives on ephemeral storage and resets on every redeploy or reboot (the
sidebar says so). No real volunteer data anywhere.

## Run locally

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\python -m streamlit run app.py
```

Requires `ANTHROPIC_API_KEY` via `.streamlit/secrets.toml` or environment
(`.env` supported locally).

## Tests

```
.venv\Scripts\python -m pytest tests/ -q
```

Measurement notes from the Phase 5 deployment run live in `NOTES.md`.
