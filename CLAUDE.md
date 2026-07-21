# CLAUDE.md — NCA Volunteer Matching (production-demo build)

Single-file Streamlit + LangGraph app being hardened, rebuilt around a
Likert scoring architecture, and deployed as a public demo. **PLAN.md is
the work order — read it before writing any code. Work its phases in
order; do not skip ahead or freelance.**

## Commands
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run: `streamlit run app.py` (needs `ANTHROPIC_API_KEY` via `.env` locally)
- Tests: `python -m pytest tests/ -q` — must be green before AND after every commit
- Quick syntax gate: `python -m py_compile app.py`

## Architecture (memorize the boundary)
LLM calls render narrow judgments: `classify_needs` (Opus 4.8 extracts
structured need sets), the Likert scorer (Haiku 4.5, four 1–5 items per
matched volunteer), and on-demand reasoning (Sonnet 4.6, per-card button,
outside the graph). Deterministic code decides everything else:
eligibility (`run_matching`, nine checks), certification policy
(`infer_mandatory_certs`), capacity/margins/counterfactuals, box collapse
(T2B/Neutral/B2B), tier assignment (thresholds + caps), ordering, and
`postprocess_recommendations` (assembly + caps + sort).
**Never move a rule into a prompt; never hardcode a judgment call.**
Graph: classify → interrupt (human skills review) → match → score →
tier-map → write record. The interrupt stays.

## Model matrix (fixed — see PLAN.md §1a; no UI model selector)
- Classifier: `claude-opus-4-8`, `thinking={"type":"adaptive"}` +
  `output_config={"effort":"medium"}`, **no temperature**, native
  structured outputs via `output_config.format` (json_schema).
- Scorer items: `claude-haiku-4-5`, temperature 0.2, native structured
  outputs, schema `{"selection": 1–5}`.
- Reasoning: `claude-sonnet-4-6`, temperature 0.2, plain text,
  `max_tokens ≈ 200`.
All calls go through the native `anthropic` SDK helpers in one module —
no LangChain LLM wrappers anywhere (`langgraph` stays for orchestration).
Never pass temperature to a thinking-enabled call. Never bind structured
output via forced tool choice on a thinking-enabled call — native
`output_config.format` only.

## Hard rules
- Safety fails closed: mandatory certs derive from the need set's work
  type (`applicable_skills ∪ confirmed_skills`), never from user checkbox
  diligence alone.
- Tier names are fixed: Perfect / Good / Technical / Almost Match. Tier
  thresholds (`PERFECT_MIN=10`, `GOOD_MIN=2`) and caps live in code
  constants — never in prompts; the model never sees the word "tier" at
  scoring time.
- Box collapse happens in code; models return raw 1–5 only. Store raw
  selections in the record — never discard the distribution.
- Reasoning never mutates a tier. "On second thought" ⇒ `dissent=true` on
  the logged event; tier unchanged; text displayed verbatim.
- A failed scorer item is never defaulted to a fake Neutral — the
  volunteer falls back to deterministic Technical Match with a note.
- Concurrency is fixed: 4 items × 4 volunteers in flight = 16 calls per
  wave; wave order is matched-list order. No tuning knobs.
- Request-record schema changes require bumping `schema_version` (born at
  2) and a migration note in the commit body. `reasoning_events` is
  append-only — never UPDATE or DELETE its rows.
- No new dependencies without a written reason in the commit message.
  stdlib `sqlite3` for persistence — no ORM.
- `langchain-openai` exists in requirements only through Phase 1 (the
  current file imports it); Phase 2 item S1 deletes it. Do not port the
  old single-call recommender to Anthropic — it is deleted, not migrated.
- Do not implement anything from PLAN.md §11 (performance) or §12
  (aesthetics). Do not restyle UI. Do not touch `app__24_.py` /
  `app__25_.py` (legacy — the higher filename number is the OLDER code).
- Secrets: `st.secrets` → env fallback. Nothing secret ever committed;
  check history before pushing.

## Testing discipline
- Every behavior change lands with a regression test in the same commit.
- Tests import the app via `importlib` and monkeypatch `load_roster` /
  `load_assignments` with fixtures. **No test calls a live API.** LLM
  configuration is asserted via mocked-transport request-body inspection
  only (model, `thinking`, `output_config`, temperature presence/absence).
- The suite includes guard tests for things that already work
  (committed-hours ISO-week math, degenerate-branch stripping,
  per-need-set scoping, `sanitize_for_state` round-trip). If one goes
  red, the change is wrong — not the test.
- Pure-code coverage is mandatory for: score→tier mapping over all nine
  attainable sums, cap interactions, wave partitioning, failure fallback,
  the dissent detector, and the gap-notes builder.

## Refactoring discipline
- Behavior commits and structure commits never mix.
- Phases 1–2: edit in place on `app.py`. Phase 4: move code verbatim into
  `core/` with zero behavior edits; the unchanged suite staying green is
  the acceptance criterion.
- Match the existing code style: section banner comments, docstrings that
  state design rationale, type hints on public functions.
- One item per commit, imperative subject referencing the PLAN item
  (e.g., `fix 3: scarcity-aware claiming`, `s4: deterministic tier caps`).

## Data
`data/` holds the roster CSV and assignments XLSX (path constants near the
top of the file — keep them pointing at `data/`). `requests.db` is
generated, gitignored, WAL-mode SQLite seeded by `data/seed_requests.py`
on first run (v2 rows + seeded reasoning events, at least one dissent).
Roster vocabulary constants (`VALID_SKILLS`, certs, languages, days,
blocks, areas) are the source of truth end-to-end — the classifier prompt,
canonicalization, matcher, and scorer context all reference them; never
let them drift apart. `LIKERT_ITEMS` and `SCORE_MAP` are likewise
source-of-truth constants; item wording changes are a spec change, not a
refactor.

## When uncertain
Prefer the smaller diff. If a change isn't in PLAN.md, isn't a test, and
isn't required to make a PLAN.md item work, don't make it — flag it in the
session summary for Ethan instead. For Anthropic API parameter questions,
re-verify against the docs pages linked in PLAN.md §1b rather than relying
on memory.
