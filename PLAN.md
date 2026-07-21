# NCA Volunteer Matching — Production-Grade Demo: Unified Plan (v2)

**Audience:** Claude Code session (with Chrome extension for the deployment phase)
**Source of truth:** `app_definitive_8.py` (v3 lineage, 2,217 lines)
**Companion file:** `CLAUDE.md` (session conventions — read it first)
**Supersedes:** PLAN.md v1 and PLAN_SCORING.md — discard both; this is the
complete plan.
**Author of record:** Ethan Pochna. Fix specs derive from a line-level audit
with an executed test harness — items marked *[executed]* were reproduced
against the real module. API mechanics in §1b were verified against live
Anthropic docs on 2026-07-21.

---

## 0. Session protocol

1. Work the phases in order. Do not begin a phase until the prior phase's
   exit criteria are met.
2. Behavior changes and structure changes are NEVER mixed in one commit.
   Phases 1–2 change behavior on the intact single file; Phase 3 proves it;
   only then does Phase 4 move code without changing behavior.
3. Every item ships in its own commit with its regression test. Suite green
   at every commit.
4. Two sections at the bottom are **reserved** (performance, aesthetics).
   Do not implement anything from them, even where it looks easy.
5. Audit fix numbers 0, 7, 8, and 9 are **intentionally absent** from
   Phase 1: they patched the old single-call recommender, which this plan
   replaces wholesale in Phase 2 before it would ever ship. Superseded
   pre-implementation; do not resurrect them.

---

## 1. Context and goal

Single-file Streamlit app implementing the **Matching Agent** of a
two-agent volunteer-coordination system for Northbridge Community Alliance
(the **Insights Agent** is designed, not built; the request record is the
bridge). Target pipeline after this plan:

    [input form] → classify_needs (Opus 4.8, LLM) → ⏸ human skills review
      → match_volunteers (deterministic, 9 checks)
      → score_volunteers (Haiku 4.5 — 4 Likert items × volunteer, parallel)
      → tier mapping + caps (deterministic)
      → write_request_record (SQLite) → tiered display
    On-demand, outside the graph: per-card "Get reasoning" → Sonnet 4.6

Orchestration: LangGraph `StateGraph` + `InMemorySaver`, human-in-the-loop
via `interrupt_before=["match_volunteers"]`. All model calls go through the
native `anthropic` SDK (§1b explains why not LangChain wrappers).

**Goal:** publish a production-grade public demo on Streamlit Community
Cloud. Production-grade = three pillars: **bugs** (Phase 1), **scoring
architecture + speed** (Phase 2), **aesthetics** (reserved for its own
session).

### 1a. Model matrix (fixed — no UI selector; D-J)

| Stage | Model | Sampling | Reasoning | Output mode |
|---|---|---|---|---|
| Classifier | `claude-opus-4-8` | temperature **unset** | `thinking={"type":"adaptive"}` + `output_config={"effort":"medium"}` | Native structured outputs: `output_config.format` = json_schema from `ClassifierOutput.model_json_schema()` |
| Scorer items (4 × volunteer) | `claude-haiku-4-5` | temperature 0.2 | none | Native structured outputs, schema `{"selection": integer enum 1–5}` |
| On-demand reasoning | `claude-sonnet-4-6` | temperature 0.2 | none | Plain text, `max_tokens ≈ 200` |

### 1b. Verified API mechanics and the one landmine

Structured outputs are GA on the Claude API for Opus 4.8 and Haiku 4.5
(`output_config.format`, grammar-constrained; tiny schemas compile
instantly). Opus 4.8 "medium reasoning" = adaptive thinking
(`thinking: {"type": "adaptive"}`) with the effort control
(`output_config: {"effort": "medium"}`); `output_config` carries `format`
and `effort` together. **Landmine:** forced tool choice (the LangChain-style
structured-output binding) is not the sanctioned path on a thinking-enabled
call — the classifier MUST use native `output_config.format`. Consequence:
native `anthropic` SDK for all three call types, Pydantic validating
responses; no `langchain-openai`, no `langchain-anthropic`; `langgraph`
stays (its nodes are plain functions). Never pass temperature to the
thinking-enabled classifier call. Re-verify parameter names in-session:
https://platform.claude.com/docs/en/build-with-claude/structured-outputs
and https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking

**Dependency sequencing note (read before Phase 0):** the current file
imports `langchain_openai`, and Phase 1's fixes land on that file with
offline tests only — so `langchain-openai` stays in requirements **through
Phase 1** purely so the module imports, and is deleted in Phase 2 item S1
when the provider migration lands. Do not port the old recommender to
Anthropic; it is deleted, not migrated.

## 2. Provenance — which file is real

- `app_definitive_8.py` — CURRENT. Base everything on this.
- `app__24_.py` — older v3. `app__25_.py` — v2, oldest despite the higher
  filename number (browser download counters, not versions).
- Never merge from or "restore" anything out of the legacy files.

## 3. Architectural invariants — violating any of these is a failed task

- **I1. The LLM/deterministic boundary.** LLMs interpret intent
  (classifier) and render narrow judgments (Likert items, reasoning prose).
  Code decides: eligibility, certification policy, capacity math, margins,
  counterfactuals, box collapse, tier assignment, caps, ordering.
- **I2. Human-in-the-loop stays.** The interrupt between extraction and
  matching is a feature.
- **I3. Safety policy is code-enforced and fails closed.** Mandatory certs
  (youth-facing → Background Check + Child Safety; food-handling → Food
  Safety; driving → Driver Authorization) fire on the need set's work type
  even when the user confirms nothing (Fix 1).
- **I4. Tiers live in code, never in prompts.** Thresholds + caps are named
  constants; the model never sees the word "tier" at scoring time. Box
  collapse (T2B/Neutral/B2B) happens in code; models return raw 1–5 only.
- **I5. Reasoning never mutates a tier.** Sonnet dissent is logged, not
  applied.
- **I6. The request record captures the full pipeline** — extraction,
  confirmations, matches, margins, counterfactuals, raw Likert selections,
  boxes, totals, caps, tiers, and reasoning events. It is the Insights
  Agent's future input.
- **I7. Tier vocabulary is fixed:** Perfect / Good / Technical / Almost
  Match.

## 4. Decision log (all locked unless marked default)

| # | Decision |
|---|----------|
| D1 | Audit fixes 1–3, 5–6, 10 approved as specified (Phase 1). |
| D2 | Fix 4: notice stays a **hard check**; add date-pair validation + notice display. |
| D3 | Fix 11: request record is **SQLite**, born at **schema_version 2** (no CSV, no v1, no migration — nothing exists yet). |
| D4 | Fix 12: `is_recurring` removed from the matcher signature; kept in form/state/record; one line added to scorer shared context when set. |
| D5 | Scorer: per volunteer, **4 Likert items** (verbatim from `plaintext_ranking_prompts.txt`), each its own Haiku 4.5 call; 5-point output collapsed in code to B2B/Neutral/T2B scored −1/+1/+3. |
| D6 | Concurrency: 4 items × **4 volunteers in flight = 16 calls per wave**; both numbers hardcoded; wave order deterministic (matched-list order). |
| D7 | Tier thresholds: **Perfect ≥ 10, Good 2–8, Technical ≤ 0** (sums are even values in [−4, 12]; ≥10 ⟺ zero B2B and ≤1 Neutral). Caps after thresholds. |
| D8 | No special-case rule for a B2B on the schedule item. |
| D9 | Reasoning is **on-demand**: per-card "Get reasoning" button → Sonnet 4.6 with tier-conditional prompts (Perfect prompt is Ethan's verbatim text; Good/Technical variants in S6 authored in his pattern, approved). |
| D10 | `configuration_notes` does not exist in this system — no prompt, no state field, no column, no banner. |
| D11 | All-Anthropic stack per §1a; classifier is Opus 4.8 medium reasoning. |
| D12 | Temperature 0.2 for Haiku and Sonnet; unset for Opus. |
| D-G (default) | Dissent: reasoning beginning "On second thought" (case-insensitive) sets `dissent=true` on the logged event; tier unchanged; text shown verbatim. |
| D-H (default) | Almost Match cards: templated blocker reasoning inline, **no** button. |
| D-I (default) | Sonnet pinned `claude-sonnet-4-6`. |
| D-J (default) | Sidebar model selector removed; read-only "Models in use" caption. |

## 5. Phase 0 — Repo scaffold

    nca-volunteer-matching/
      app.py                  ← app_definitive_8.py, renamed
      core/                   ← empty until Phase 4
      data/
        northbridge_volunteer_roster.csv
        northbridge_volunteer_assignments.xlsx
        seed_requests.py      ← builds v2 demo request history (S7)
      tests/
        test_regressions.py
        fixtures.py
      requirements.txt        ← includes langchain-openai THROUGH Phase 1 only (§1b note)
      runtime / python pin    ← per current Streamlit Cloud docs (verify, don't assume)
      .streamlit/             ← empty placeholder; theme is aesthetics-pass material
      .gitignore              ← .env, __pycache__/, *.pyc, requests.db, .streamlit/secrets.toml
      PLAN.md  CLAUDE.md  README.md

- Update `ROSTER_PATH` / `ASSIGNMENTS_PATH` constants to `data/…`.
- Requirements pins verified so far: pandas 3.x (module runs under 3.0.2);
  `anthropic` and the rest pin at install-time resolution, lockfile recorded.
- Keys: `ANTHROPIC_API_KEY` via `st.secrets` → env fallback; `.env` local
  only. (`OPENAI_API_KEY` exists only until S1 deletes the old call sites.)

**Exit:** venv builds; `streamlit run app.py` reaches the input form locally.

## 6. Phase 1 — Correctness fixes (in-place on `app.py`)

Anchors: function names first, `app_definitive_8.py` line numbers second
(lines drift — trust the function). All tests here are offline (monkeypatched
loaders; no live API).

### Fix 1 — Mandatory certs fire on work type, not checkbox diligence *[executed]*
`infer_mandatory_certs(confirmed_skills)` inside `run_matching` (~1040) plus
the unchecked-by-default UI means zero confirmed skills ⇒ zero policy certs
⇒ a volunteer with **no background check** matched a youth tutoring request
in testing. Change the derivation basis:

```python
policy_basis = set(confirmed_skills) | set(need_set.get("applicable_skills", []))
mandatory_certs = infer_mandatory_certs(sorted(policy_basis))
```

Mirror in the review UI (~1973): displayed auto-added certs computed from
`extracted ∪ confirmed`, caption reworded to "based on the type of work
identified." Accepted trade-off: inclusive classifier suggestions can
over-trigger certs; review-step visibility is the mitigation.
*Test:* NoCheck Nate (no certs) vs Cleared Clara; `confirmed_skills=[]` +
`applicable_skills=["Tutoring - Math"]` ⇒ only Clara matches.

### Fix 2 — OR-branch subsumption in `normalize_flexible_requirement` (~843) *[executed]*
A branch emptied by AND-subsumption means the OR clause is already satisfied
whenever AND holds; dropping only the branch leaves sibling branches
mandatory and the requirement strictly harder. Executed proof:
`AND=["Mon"], OR=[["Mon"],["Mon","Sat"]]` (≡ "Monday") normalized to
Monday-AND-Saturday; a Monday-only volunteer was rejected. Distinguish the
two empty cases:

```python
canonical = [unique canonicalized values of branch]   # garbage removed
if not canonical:
    continue                      # garbage-only branch → drop the branch
remaining = [v for v in canonical if v not in and_set]
if not remaining:
    or_branches = []              # branch implied by AND → whole OR is vacuous
    break
# then existing dedupe on tuple(remaining) and append
```

*Tests:* subsumed case passes a Monday-only volunteer; garbage-only branch
(`[""], ["NA"]`) drops without vacating a real sibling; plain cases
unchanged.

### Fix 3 — Scarcity-aware claiming in `match_volunteers_node` (~1414) *[executed]*
Claiming `ns_matched[:count]` in roster order strands feasible assignments
(executed: the only Spanish+Car volunteer was claimed by the Spanish slot;
the driver slot went unfilled though Bea→intake, Ana→driver existed).
Remedy: **Pass 1** computes each need set's matched pool against the FULL
roster (reuse `run_matching`, keep only `matched`); **Pass 2** keeps today's
sequential depletion loop but claims by scarcity —
`sorted(ns_matched, key=lambda v: (count of OTHER full pools containing v,
ns_matched.index(v)))[:count]`. Tie-break = pool order (deterministic).
*Tests:* two-slot fixture fills both slots; tie fixture claims in pool
order; single-need-set behavior byte-identical.

### Fix 4 — Date-pair validation + notice visibility *[executed]*
Notification date after target date drives notice negative and silently
blocks the whole roster on "Notice Period." The check stays hard
(volunteer-stated minimum notice is a commitment; near-misses already
surface as Almost Match). UI-side changes: submit handler errors and aborts
when `has_specific_date` and `notification_date > target_date`; warns when
`notification_date < today` (backdating inflates notice); review stage
displays "Notice window: N day(s)."
*Test:* matcher-level executed case retained as failure-mode documentation;
UI guard verified by inspection + golden scenario G4.

### Fix 5 — Margins: per-group storage, first-wins flat view (~1446) *[executed]*
`all_margins.update(...)` keyed by volunteer ID lets a later need set
overwrite an earlier one's margins (executed: Spanish shown as an "extra"
language against the need set that REQUIRED Spanish). Remedy: attach
`"margins": match_result["margins"]` inside each match-group dict (rides
inside `matched_volunteers_json` — no new column); build the flat display
dict with `setdefault` so the FIRST (most-constrained) need set wins.
*Test:* nested margins present per group; flat
`margins["C"]["extra_languages"]` excludes Spanish.

### Fix 6 — Word-boundary soft-preference detector (~977) *[executed]*
Substring matching double-fires ("prefer monday" hits `mon` and `monday`)
and false-positives ("prefer monetary donations"). One regex per canonical
value:

```python
signal = r"(?:prefer(?:red|s)?|ideally)"
rf"\b{signal}\s+(?:{'|'.join(map(re.escape, aliases))})s?\b"   # per day
rf"\b{signal}\s+{block.lower()}s?\b"                            # per block
```

These violations feed the Phase 2 tier caps (S4).
*Tests:* "We prefer monday sessions" → one violation; "prefer monetary
donations" → zero; "prefers mondays" → one.

### Fix 10 — Skills get fuzzy canonicalization like every other domain
A classifier emitting "tutoring - math" is silently dropped from the
confirmation UI (~1319) — never confirmable, never enforced, never
cert-triggering. Add `"skills": VALID_SKILLS` to `canonicalize_value`'s
fallback map (~405); route the classifier's skill collection and per-need-set
`applicable_skills` sanitization through it; pass `domain="skills"` on the
roster-side parses in `check_skills` and the margins block.
*Test:* lowercase "tutoring - math" survives into `extracted_skills`;
roster "PANTRY OPERATIONS" satisfies "Pantry Operations."

### Fix 12 — `is_recurring`: stop pretending, start using
Remove the never-read parameter from `run_matching`'s signature. Keep the
field in form/state/record. Its use point is now Phase 2: when set, the
scorer's shared context (S3) gains one line — "This is a recurring need
through {recurring_end_date}; weigh sustained availability."
*Test:* signature change compiles; context line appears when flag set
(asserted in S3's prompt-factory tests).

**Phase 1 exit:** fixes committed individually; suite green;
`python -m py_compile app.py` clean.

## 7. Phase 2 — Scoring & reasoning rebuild

This phase deletes the single-call recommender and its schemas
(`RecommenderOutput`, `VolunteerRecommendation`,
`RECOMMENDER_SYSTEM_PROMPT`) and replaces them with the Likert scorer,
deterministic tiering, and on-demand reasoning. The audit's postprocessor
findings resolve structurally here: hallucinated IDs and duplicates are
impossible (code supplies IDs, one result slot per volunteer); tier-forcing
and its inverse are moot (code assigns tiers); backfill is reborn as the
failure fallback (S3); ordering is a plain sort (S4).

### S1 — Provider migration and call helpers
One shared `anthropic.Anthropic()` client; three helpers per §1a:
`call_classifier(prompt_ctx) -> ClassifierOutput` (Opus 4.8, adaptive
thinking, effort medium, native json_schema output; port
`classify_needs_node` to it), `call_likert_item(shared_ctx, profile, item)
-> int`, `call_reasoning(bundle, tier) -> str`. Delete all
`langchain_openai` imports and the old `ChatOpenAI` call sites; requirements
drop `langchain-openai`, add `anthropic`; secrets story moves to
`ANTHROPIC_API_KEY`; sidebar becomes the D-J read-only caption.
*Tests:* mocked-transport request-construction asserts per stage — body
contains the right model, `thinking`, `output_config` (`effort`, `format`),
and temperature presence/absence per §1a. No test calls the live API.

### S2 — Likert item constants
`LIKERT_ITEMS`: hardcoded 4-tuple with each question's verbatim text from
`plaintext_ranking_prompts.txt` (Q1 overall fit, Q2 schedule friction, Q3
willingness, Q4 recommendation) + anchor labels (5=Strongly agree …
1=Strongly disagree). `SCORE_MAP = {5: 3, 4: 3, 3: 1, 2: -1, 1: -1}`. Box
collapse in code only; the model returns raw 1–5.
*Tests:* score-map totality; item texts match the source file.

### S3 — Scoring node (replaces `recommend_node`)
Prompt factory: shared context = original request, need-set description,
**stated** soft preferences, unconfirmed suggested skills labeled
"suggested but not required — context only" (the surviving intent of old
audit fix 9), and the recurring line when set; + one compressed volunteer
profile (same fields as before — no capacity numbers, history, or home
area); + one item's question and anchors. Execution: volunteers in
matched-list order, fixed waves of 4 (`VOLUNTEERS_IN_FLIGHT = 4`); per wave,
16 calls in one `ThreadPoolExecutor(max_workers=16)`; group by volunteer,
sum. Failure policy: one retry with jitter per item; persistent failure ⇒
the volunteer is NOT scored with a defaulted value (a fake Neutral skews the
sum) — deterministic **Technical Match** fallback with a templated
"scoring unavailable" note. Almost-matched volunteers never enter the
scorer; empty matched pool ⇒ zero calls (skip-on-empty is structural).
*Tests:* wave partitioning deterministic for 1/4/5/9 volunteers;
aggregation; single-item failure fallback isolated to its volunteer;
almosts excluded; recurring + suggestions lines present when applicable.

### S4 — Deterministic tier mapping and caps
`PERFECT_MIN = 10`, `GOOD_MIN = 2`: sum ≥ 10 → Perfect; 2–8 → Good; ≤ 0 →
Technical. Caps after, in order: (a) stated-soft-preference violation → max
Good; (b) Fix-6 regex schedule violation → max Good. Nothing else (D8).
Almost Match assigned upstream, never by score. `postprocess_recommendations`
persists as assembly + caps + sort (tier rank, then name) — deck-name
parity, now honest.
*Tests:* full table over all nine attainable sums; each cap alone and
together; caps cannot promote; sort stability.

### S5 — Result assembly, state, gap notes
Result dicts keep the display-compatible outer shape
`{volunteer_id, tier, reasoning: ""}` plus `raw_selections`, `boxes`,
`total_score`, `caps_applied`. Almost entries carry templated blocker
reasoning inline. `gap_notes` = deterministic builder (per need set:
"need N, found M" + top counterfactual blockers with counts). All
`configuration_notes` code paths deleted (D10).
*Tests:* gap-notes builder on the Fix-3 fixture; state round-trips through
`sanitize_for_state`; display renders unchanged tier grouping.

### S6 — On-demand reasoning ("Get reasoning")
Perfect/Good/Technical cards render the button. Click ⇒ one Sonnet 4.6 call:
bundle = request summary, need-set description, stated soft preferences,
the volunteer's compressed profile, assigned tier, and the item results
(raw selections + boxes + total — the model explains the tier from its
evidence) + the tier-conditional prompt. Cache in `st.session_state` keyed
(thread_id, volunteer_id); render in the caption slot; log per S7.

Perfect (Ethan's verbatim):

> Review why this respondent is a perfect fit for the request, rather than
> just a good fit. If you believe they are not a perfect fit, preface your
> reasoning with, "On second thought...". Explain their fit by highlighting
> how their profile aligns with the request in 1-2 sentences.

Good (approved):

> Review why this respondent is a good fit for the request, rather than a
> perfect fit or a merely technical one. If you believe this tier is wrong
> in either direction, preface your reasoning with, "On second thought...".
> Explain their fit by highlighting how their profile aligns with the
> request, and what keeps them short of a perfect fit, in 1-2 sentences.

Technical (approved):

> Review why this respondent technically qualifies for the request but may
> not be a natural fit. If you believe they are a stronger fit than a
> technical match, preface your reasoning with, "On second thought...".
> Explain what qualifies them and where the misalignment lies in 1-2
> sentences.

Dissent per D-G: leading "On second thought" (case-insensitive, tolerant of
straight/curly apostrophes and trailing punctuation) ⇒ `dissent=true` on the
logged event; tier unchanged; text verbatim. Almost cards: no button (D-H).
*Tests:* dissent detector (positive; negative; mid-text mention is NOT
dissent); cache prevents duplicate calls on rerun; bundle contains scores.

### S7 — Persistence, born at schema_version 2
SQLite via stdlib `sqlite3`, file `requests.db` (gitignored), WAL mode,
writes in transactions. Table `requests`: the existing record columns with
`request_id = str(uuid.uuid4())` (full), `schema_version` default 2,
`recommendations_json` entries carrying
`{volunteer_id, tier, raw_selections, boxes, total_score, caps_applied}`,
deterministic `gap_notes`, no `configuration_notes` column. Append-only
table `reasoning_events(request_id, volunteer_id, tier, model, text,
dissent INTEGER, created_at)` — one row per button fetch; dissent rate
becomes a queryable QA metric for the Insights Agent.
`data/seed_requests.py`: ~30 plausible v2 rows + a handful of seeded
reasoning events (at least one dissent), invoked at startup when
`requests.db` is absent; UI caption "Demo dataset — resets on redeploy."
*Tests:* v2 round-trip via `pd.read_sql` incl. raw selections; concurrent
writes (requests and reasoning events) both land; seed idempotent; dissent
stored 0/1; full-UUID format.

**Phase 2 exit:** old recommender fully deleted (grep for
`RECOMMENDER_SYSTEM_PROMPT`, `RecommenderOutput`, `ChatOpenAI` returns
nothing); suite green; a manual local run scores a request end-to-end and
fetches reasoning for one card per tier.

## 8. Phase 3 — Regression suite consolidation (`tests/`)

Harness pattern (proven): import `app.py` via `importlib`, monkeypatch
`load_roster`/`load_assignments` with fixtures, call functions directly;
LLM behavior asserted only via mocked-transport request bodies. Fixtures:
NoCheck-Nate/Cleared-Clara (Fix 1), Ana/Bea two-slot (Fix 3), Ana/Cai
dual-match (Fix 5), the messy 5-row assignments frame (committed-hours
no-regression: exactly 5.0 for the 2026-07-23 ISO week).

Permanent guard tests for things that already work (if one goes red, the
change is wrong, not the test): ISO-week committed-hours math incl. status
normalization and unparseable dates; degenerate OR branches stripped before
evaluation; per-need-set skill scoping; `sanitize_for_state` round-trip.

**Exit:** one `pytest` run covers Phases 1–2 and the guards; documented
green run.

## 9. Phase 4 — Structure-only modularization (behavior-frozen)

Mechanically split `app.py` → `core/schemas.py` (Pydantic + GraphState),
`core/matching.py` (canonicalization, flexible requirements, `run_matching`,
margins/history), `core/policy.py` (vocab constants, cert rules),
`core/scoring.py` (Likert constants, prompt factory, waves, tier mapping,
caps, `postprocess_recommendations`), `core/reasoning.py` (tier prompts,
bundle builder, dissent detector), `core/llm.py` (client + three call
helpers), `core/records.py` (SQLite), `core/graph.py`, `app.py` (UI only).
Move code verbatim — zero behavior edits; the unchanged suite staying green
is the acceptance criterion.

## 10. Phase 5 — Deployment (Chrome-extension territory)

1. Push to GitHub; verify no secrets in history.
2. Streamlit Community Cloud: app from repo; `ANTHROPIC_API_KEY` in
   dashboard secrets; follow CURRENT Streamlit Cloud docs for the python
   pin — do not assume a mechanism from memory.
3. **Browser acceptance script** against the LIVE app, in order:
   - G1 *Cert enforcement:* "I need a math tutor for kids on Thursday
     afternoon," confirm NO skills → every recommended volunteer holds
     Background Check - Cleared + Child Safety Training - Completed; review
     screen showed the auto-added certs.
   - G2 *OR-logic:* a request whose day constraint reduces to "Monday (or
     Monday and Saturday)" → a Monday-only volunteer appears.
   - G3 *Two-slot allocation:* "one Spanish-speaking intake volunteer and
     one delivery driver" → both slots fill when jointly feasible.
   - G4 *Date guard:* notification date after target → blocked at submit;
     fixed dates → notice window visible on review.
   - G5 *Deterministic tiering:* tiers correspond to the stored score data;
     a request with an explicit "preferably …" shows the stated-preference
     cap (violators max at Good); a volunteer missing only unconfirmed
     suggested skills can still reach Perfect.
   - G6 *Reasoning flow:* click "Get reasoning" on one card per tier —
     1–2 sentences render, a `reasoning_events` row lands (query the DB);
     any response opening "On second thought" carries `dissent = 1`.
   Screenshot each; any failure returns to the owning phase with a new test.
4. README: live URL, one-paragraph pitch, demo-dataset note.
5. **Measurement duties** (record in README or a NOTES file): one-wave
   scorer latency vs nothing-to-compare baseline noted; Opus classifier
   latency at medium effort; distribution of raw Likert selections over the
   first ~50 requests (watch ceiling clustering / central tendency at
   temperature 0.2 — if every volunteer sums to 12, item anchors need
   hardening, and the record will show it); dissent rate.

**Exit / definition of done:** G1–G6 pass on the deployed URL; suite green
in a documented run; no reserved-section code present; PLAN phases checked
off in the final commit message.

## 11. RESERVED — performance backlog (DO NOT IMPLEMENT)
Matcher vectorization and committed-hours precompute (measured 14.8s at
2,000 volunteers / 20,000 assignments vs 101ms at demo scale — real only at
scale); classifier memoization for repeated demo prompts; progressive
per-wave rendering of tier results (low value at 1–2 waves; revisit if
rosters grow); Anthropic Batch API for future offline Insights re-scoring
only — never the interactive path.

## 12. RESERVED — aesthetics backlog (DO NOT IMPLEMENT)
The tier-color `<div>` wrappers don't actually wrap Streamlit-native
children (renderer closes them immediately) — card styling is currently a
no-op strip. Also: theme, typography, layout polish,
`.streamlit/config.toml`, score-box chips on cards, a visual dissent
marker. Separate session.
