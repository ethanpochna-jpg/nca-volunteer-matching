# NCA Volunteer Matching

Matching Agent of a two-agent volunteer-coordination system for Northbridge
Community Alliance. Streamlit + LangGraph, all-Anthropic model stack.

Status: rebuild in progress per PLAN.md (Phases 0–5). Live URL, pitch, and
demo-dataset notes land here at Phase 5.

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
