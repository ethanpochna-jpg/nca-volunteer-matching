# NOTES — Phase 5 measurement duties (PLAN §10.5)

Recorded 2026-07-22 against the live deployment
(https://nca-volunteer-matching.streamlit.app, Python 3.13.14, streamlit
1.60.0). Method: wall-clock observation during the G1–G6 browser acceptance
run. Community Cloud offers no shell, so the request-record DB was not
queried directly; distributions below are read off displayed tiers, whose
mapping from raw selections is deterministic and covered by pure-code tests.

## Latency (observed, single-user)

- **Opus classifier, adaptive thinking at medium effort:** review screen
  rendered within ~5 s on a short request and within ~15–20 s on multi-need
  requests. No timeouts, no retries observed.
- **Haiku scorer waves** (4 items × 4 volunteers in flight, 16 calls/wave):
  results rendered ~30 s after confirm for a 5-volunteer pool (2 waves) and
  ~40 s for a 19-volunteer pool (5 waves) — roughly 8–10 s per wave
  end-to-end including tier mapping and the record write.
- **Sonnet reasoning:** each on-demand fetch rendered in well under 8 s.

## Raw-selection distribution (floor effect — confirmed, not tuned)

The strong floor effect first seen in the Phase 4 exit run reproduces on the
live app: when a stated soft preference is unmet, all four items tend to
bottom-two-box, sending the sum to the floor. Live tier outcomes today:

- Pantry-on-Monday (all candidates violated something): 0 recommended.
- Office data entry, Monday-or-Monday-and-Saturday: 1 Perfect / 1 Good /
  3 Technical — the mid-band is reachable (a Good at sum ∈ [2, 9]).
- Spanish intake + delivery driver: 1 Perfect + 1 Good across two need sets.
- Data entry "preferably Fridays" (19 scored): 2 Perfect / 0 Good /
  17 Technical — bimodal, floor-dominated.

Distribution is bimodal-leaning ({floor, 12}) but not strictly binary; Good
appeared in 2 of 4 scored requests. Because sums rarely land in [10, 12) for
preference violators, the S4 stated-preference cap seldom binds — violators
are already at the floor. Item-anchor hardening would change this, but item
wording is a spec change (CLAUDE.md); decision is Ethan's. Record only.

## Dissent rate

0 of 7 live reasoning fetches opened with "On second thought" (3 tiers
covered, including 4 borderline Technical cards chosen to provoke it). Local
Phase 2/4 runs: 0 of 3. Observed live dissent rate to date: 0/10. The dissent
path itself is exercised by the seeded `reasoning_events` row (dissent=1) and
the pure-code detector tests.

## Deployment verification notes

- Cloud build log shows `Using Python 3.13.14` (pinned via the deploy-time
  Advanced settings dropdown — the only mechanism current Cloud docs honor;
  `runtime.txt` is ignored).
- Secrets: `st.secrets → env` fallback works on Cloud; note the client is a
  lazy singleton (`core/llm.py:get_anthropic_client`), so a secret saved
  after first boot requires an app reboot to take effect — this bit us once
  at first deploy ("Could not resolve authentication method").
- `requests.db` seed-on-absent survived a reboot (ephemeral storage wiped,
  app reseeded cleanly); no `database is locked` errors observed under
  single-user WAL usage.
