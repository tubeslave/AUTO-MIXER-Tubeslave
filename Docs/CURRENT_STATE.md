# Current State

Generated: 2026-06-19T12:40:32Z

## Repository snapshot

- Checkout: `/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main`
- Branch at latest inspection: `codex/automixer-autopilot-control-plane`
- Latest verified product/runtime commit:
  `2f8bb432 Surface proof plan summary flags in no-WING markdown`
- Tracked/untracked worktree state: dirty before this setup, with many
  unrelated modified backend/tests/docs files and generated artifacts.
- Complete tracked file count from `rg --files`: 13160.
- Existing root agent file: only `./AGENTS.md`.
- Canonical docs tree: `Docs/`. On this macOS checkout `docs` resolves to the
  same tree, but new durable files are written as `Docs/...`.

## Verified repo facts

- `AGENTS.md`, `CLAUDE.md`, `Docs/CONVENTIONS.md`, `Docs/ARCHITECTURE.md` and
  `Docs/TOOLS.md` define the repo safety and architecture canon.
- Long-term project goal exists in `.ai/project-layer/automixer_goal.md`.
- Current execution priority is `Docs/MVP_WING_SUPERVISED.md` plus
  `.ai/project-layer/wing_mvp_blockers.md`.
- Project Director and direction guard are planning/audit only:
  `tools/codex_autoloop/project_director_agent.py` and
  `tools/codex_autoloop/codex_direction_guard.py`.
- Local model budget guard exists:
  `tools/codex_autoloop/check_model_budget.py` and
  `tools/codex_autoloop/model_budget_profile.json`.
- Existing CLI autoloop lock exists in `tools/codex_autoloop/run_codex_cycle.sh`
  as `.codex_autoloop.lock`.

## Current capability boundary

- Analysis and assisted proposal flows exist across many modules.
- The durable WING MVP boundary remains supervised read-only proof plus
  supervised `fader`/`gain` proof, rollback, emergency stop and evidence.
- `backend/operator_mode_policy.py` has a broader internal
  `CURRENT_PROVEN_LIVE_WRITE_KINDS`, but WING deployment live writes are limited
  by `WING_DEPLOYMENT_LIVE_WRITE_KINDS = ("fader", "gain")`.
- Full Autonomous Mixing Mode is not ready.
- No real hardware evidence was produced during this setup run.
- M001 autonomous pre-hardware work is complete and waiting for external
  WING/operator validation. The exact real-run handoff is
  `Docs/reports/AUTOMIXER_AUTOPILOT_M001_CP04_WING_READONLY_HANDOFF_20260618.md`.
- After M001-CP05, verified no-WING safety-scope hardening commits narrowed
  WING report/executor/monitor surfaces to the supervised deployment boundary:
  `2b25d91f`, `12f56180`, `56329600`, `2464f0d8`, `d20f7d46`, and
  `bb5816d4`, `192d6bff`, `ce0d3d00`, `7796d28a`, `b0c9fc00`, and
  `12ee8214`, `b0c71460`, `d84b4a63`, `7a616805`, `93e66009`, and
  `c35947b0`, `367e67bb`, `bd4e6ef0`, `e1e31ee0`, `ffddb646`,
  `8e85cf39`, `124c1115`, `cb2301e8`, `fe111f12`, `ec189179`, `cda73c11`,
  `14124ff6`, `b239d6f5`, `b595cf00`, `58196c1b`, `305bfac7`,
  `ff8a70ff`, and `2f8bb432`.
- The autonomy evidence overlay now emits and accepts only WING deployment
  live-write targets (`fader`/`gain`) for current WING autonomy-overlay
  evidence; `pan` remains outside the current WING deployment overlay boundary.
- Operator-facing product state now filters and recomputes autonomy-overlay
  rows, status counts and target buckets against the same WING deployment
  boundary before exposing supervised proof evidence to the UI.
- Supervised promotion evidence now creates current review rows from the same
  WING deployment boundary, so `pan` proof artifacts cannot become current WING
  promotion-evidence rows without a separate hardware-backed expansion.
- The supervised proof artifact evaluator itself now reports and accepts only
  the WING deployment proof types (`fader`/`gain`) for current WING proof
  evaluation; `pan` artifacts are blocked as unsupported before they can become
  dry-run-ready or promotion candidates.
- Operator proposal sequence execution audit and audit-policy gates now compare
  evidence rows and safety scopes against the active safety surface. On WING
  deployment this blocks `pan` audit rows and forged `fader/gain/pan` safety
  envelopes; offline/non-WING audits still retain the broader current-proven
  scope.
- Operator recommendation bridge cache-freshness gating now uses the same WING
  deployment apply boundary, so future recommendation imports treat `pan` as
  unsupported for current WING apply scope even when read-only pan cache exists.
- Shadow-only gain ride candidates now publish the same WING deployment
  `fader`/`gain` boundary in their safety metadata while still denying live
  promotion and live apply.
- Supervised sequence policy memory index and trust-review validators now use
  the same WING deployment `fader`/`gain` boundary, so forged or append-only
  `pan` sequence memory cannot enter current fader/gain sequence review.
- Operator-facing product-state sequence plan, dry-run, memory-draft and
  memory-writer validators now use the active safety surface, so WING blocks
  `pan` sequence-policy evidence while non-WING/future surfaces can still keep
  the broader current-proven scope.
- The no-WING release-check Markdown summary now renders
  `unexpected_supervised_live_write_kinds` beside the policy and current WING
  live-write kind fields, so operator-facing evidence does not hide a computed
  kind-widening signal.
- The WING read-only preflight decision review report now surfaces
  accepted-candidate request-sequence contract validity, source ids, and
  per-source violations in both JSON and Markdown. Accepted readiness remains
  blocked unless the accepted source ids have a valid request-sequence trace.
- The same decision review report now also blocks `accepted_decision_ready`
  until accepted source ids have complete contact trace and expected-artifact
  trace evidence, including expected-artifact-bound source ids visible in JSON
  and Markdown.
- The decision review Markdown `Blockers` section now also surfaces the
  computed request-sequence, contact-trace and expected-artifact-trace blockers
  that already keep accepted readiness closed, so operator-facing Markdown no
  longer hides those computed blockers.
- The no-WING release gate now independently checks that accepted review
  candidates missing request-sequence, contact-trace or expected-artifact-trace
  proof expose matching blockers before any readiness candidate can proceed.
- The no-WING release gate now also independently checks that accepted review
  candidates missing artifact-manifest verifier WebSocket proof expose the
  matching blocker before any readiness candidate can proceed.
- The no-WING release gate now also fails accepted review candidates whose
  request-sequence contract claims validity for source ids that do not exactly
  match the accepted source ids, blocking stale or cross-bound source traces
  from supporting readiness.
- The no-WING release gate now also fails accepted review candidates whose
  valid artifact-root contract reports roots outside the expected
  `artifacts/wing_readonly_preflight` root for the accepted source ids,
  blocking stale or cross-bound artifact roots from supporting readiness.
- The no-WING release gate now also fails accepted review candidates whose
  expected-summary artifact trace points to a different summary path than the
  accepted source id, blocking stale summary artifacts from supporting
  readiness even when summary match flags are forged true.
- The no-WING release gate now also fails accepted review candidates whose
  expected-events artifact trace points to a different events path than the
  accepted summary source, blocking stale events artifacts from supporting
  readiness even when events match flags are forged true.
- The no-WING release gate now also fails accepted review candidates whose
  request-sequence trace does not match the canonical WING read-only preflight
  request sequence, blocking stale or truncated request plans even when the
  candidate claims `request_sequence_matches_plan: true`.
- The no-WING proof-plan surfaces now also expose
  `readonly_preflight_review_decision_accepted_candidate_expected_request_sequence_by_source`,
  so accepted-candidate actual and plan request sequences are compared against
  an operator-visible canonical expected sequence map before any readiness
  candidate can proceed.
- The no-WING proof-plan surfaces now also expose
  `readonly_preflight_review_decision_accepted_candidate_expected_contacts_backend_by_source`
  and
  `readonly_preflight_review_decision_accepted_candidate_expected_queries_console_by_source`,
  so accepted-candidate contact evidence has an operator-visible expected map
  before any readiness candidate can proceed.
- The no-WING compact review-decision ledger summary now also exposes accepted
  candidate expected/actual contact maps, so summary consumers can see the
  per-source `contacts_backend` and `queries_console` requirements instead of
  only source ids and completeness booleans.
- The no-WING compact review-decision ledger summary now also exposes accepted
  candidate request-sequence trace source ids, actual/plan/expected
  request-sequence maps, per-source plan-match flags, missing source ids and
  trace completeness, so summary consumers can see the canonical request order
  proof instead of only contract validity and source ids.
- The no-WING compact review-decision ledger summary now also exposes accepted
  candidate expected-artifact trace maps, including expected artifacts,
  expected summary/events artifact paths, per-source path-match flags, missing
  source ids and trace completeness, so summary consumers can see stale or
  forged artifact bindings without reading only the detailed proof plan.
- The no-WING Markdown summary now also renders the accepted-candidate
  summary-surface required and denial flags for request-sequence,
  contact-trace and expected-artifact gates, so operator-facing Markdown can be
  checked against the same summary keys exposed in JSON.
- The no-WING Markdown summary now also renders backend prestart
  summary-surface required/readiness/live-authority flags for the prestart
  gate, prestart summary, failure-detail surface and authority-denial surface,
  so the operator-facing report exposes the same pre-backend safety gates as
  JSON.
- The no-WING Markdown summary now also renders WING read-only handoff packet
  summary-surface required/readiness/live-authority flags for packet safety,
  expected artifacts, command binding, operator sequence, operator sequence
  step gate and supervised proof-plan fields. The fresh no-WING bundle exposes
  all `33` `wing_readonly_*summary_surface*` JSON keys in `summary.md`.
- The no-WING Markdown summary now also renders next WING proof-plan
  candidate-contract and required-fields summary-surface required/readiness/live
  authority flags, plus the Project Director selected-action summary-surface
  required flag. The fresh no-WING bundle exposes all `90` `summary_surface`
  JSON keys in `summary.md`.

## Unresolved unknowns

- Exact current validity of every generated no-WING artifact under
  `artifacts/no_wing_release_check/` is not assumed.
- Current WING Rack availability is UNKNOWN.
- Current native Goal transition availability inside future scheduled
  heartbeat wake-ups must be checked at wake-up time.
- Existing large uncommitted product changes may affect future focused work,
  but the latest verified full-suite run for `2f8bb432` passed with
  `4061 passed, 1 skipped, 1 warning`.
