# Autopilot Blockers

## Current blockers

> M110/M111 current blocker (2026-07-14): CP00-CP05 are complete under the
> source-bound marker contract and the clean detached launch-only probe passed
> on commit `5c054eb9e28d29601d974850b8874858b3fe0881`.
> It contacted only the loopback backend; WING, OSC/UDP, audio stream and live
> writes remained closed. Passive CoreAudio and macOS USB enumeration found
> zero physical WING devices, while Dante Virtual Soundcard was visible but is
> not an accepted substitute. HW01 is therefore blocked before start on
> `m110_wing_usb_coreaudio_device_missing`; HW02-HW05 remain pending. Reset and
> correction counts remain zero. CP05 is dynamically checked against the
> current non-marker tree and must pass before hardware work. M005 behavior is closed;
> M006 permission is one attended-session input, not standing authority. Reset
> remains the 64-address CH1-16 bypass plus exact M006 preimage restoration
> (23/14/14 operations, 33/63/14 scope, 19/52/6 changed, 77 unique changed
> addresses), not a factory reset. Strict live stays `88/100`, allowlist `0`,
> authority `false`, target `95/100` unmet. M111 provides only an offline
> migration route; it remains blocked on a fresh run/source-bound challenge,
> exact same-console serial attestation, clean committed replay and explicit
> migration-bound M006 decision.

> Read-only decision-ledger addendum (2026-07-14): two later valid candidates
> remain `operator_review_required`. The older exact source keeps its accepted
> record, but the later candidates inherit no authority and do not reopen
> closed M005 behavior.

> M109-CP01 closure (2026-07-13): committed source checkpoint
> `b3f7d05ac1f0a2ddc9812e4723a2b3f130fefa6a` passes with `6368 passed,
> 1 skipped in 382.42s`; the full no-WING module passes `457/457` and WING
> truth rails pass `135/135`. Four positive tests are now artifact-history-
> hermetic. Production validation and physical artifacts are unchanged, and
> the canonical 13/6/7 aggregate still fails closed. Local engineering is
> restored to `95/100`; strict real-live/public readiness remains `88/100`.
> Normal allowlist remains `0`; status is `soundcheck_incomplete`; reset and
> applied-correction counts remain `0`; live authority remains denied. The
> older ledger-backed checkpoint below remains the previous full handoff; this
> block is the newer CP01 source-commit binding.

> Soundcheck P1 closure (2026-07-12): the local staged execution kernel is
> `96/100`; full-project local engineering remains `95/100`. Strict real-live
> soundcheck readiness is `88/100`, not `95/100`. Remaining blockers are an
> exactly empty normal live allowlist, no current-tree physical staged-child
> completion, no real `safe_core_ready`/`show_mix_ready` proof, unmeasured WING
> latency for the two-pass source-path sweep, and operator-review-only EQ
> boost/compressor makeup/gain-up/send-up substitutions. M005 behavior is
> closed and M006 owner intent is recorded; current-source provenance and
> technical promotion remain fail-closed. No live authority changed.

> Final architecture review (2026-07-12): the staged reanalysis lifecycle is
> now candidate/commit/discard with exact accepted-bundle identity,
> derivation+monitor+console-writer fencing, transactional observation shadow
> and cancellation-safe cleanup. No remaining P0/P1/P2 was found in this local
> scope. These fixes do not change the external strict-live blockers or grant
> WING authority.

> Superseding update (2026-07-12): local engineering readiness is `95/100`;
> strict public-release readiness remains `88/100`. M005
> disconnect/timeout/stale-state behavior is closed by the passed physical
> artifact and owner confirmation. The narrow M006 owner decision is recorded,
> while technical promotion remains blocked with
> `reason=m006_owner_decision_failed_closed` on current-source provenance.
> Standing supervised live-write kinds, live-allowed safety rows and
> release-ready live families remain empty.
> Expander, generic dynamics/de-esser, FX model/insert/return, group, matrix,
> output processing, routing and scene/preset are also unproven/proposal-only.
> The latest local Electron 43 arm64 packaged-runtime smoke passed without WING
> contact, but it is dirty-tree-bound, ad-hoc signed, `releaseEligible=false`
> and not public-release evidence. It binds all 360 app and 345 backend Mach-O
> files to arm64, passes all 18 challenged identity checks, starts the backend
> in 5207/20000 ms and preserves all four
> canonical stores. It remains below the public-release `95/100` gate.

Current blocker from `STATE.json`: M110 remains blocked before HW01 because passive CoreAudio and macOS USB enumeration found no physical WING device; Dante Virtual Soundcard 32x32 is not an accepted substitute for WING 48x48 at 48 kHz. No capture stream or OSC/UDP path was opened, and reset/corrections remain 0. M111 now provides a truthful M005 migration route without changing native M005 provenance, but activation still requires the exact WING Rack serial and same-console owner attestation, a clean committed compatibility replay, and an explicit migration-bound M006 decision. Strict real-live readiness stays 88/100: normal allowlist 0, standing authority false, safe_core_ready/show_mix_ready unproven.

- The 2026-07-11 M005 artifact is accepted as behavioral closure but is not
  current-source public-release provenance. Do not repeat the physical cable
  pull unless the owner explicitly requests it.
- Current M006 owner verification recomputes the packet from the accepted M005
  input; the recorded owner decision cannot promote readiness until that
  provenance passes.

## Current external validation requirements

- Physical WING USB/CoreAudio device named WING with 48 input and 48 output channels at 48 kHz; Dante Virtual Soundcard is not an accepted substitute
- Operator-present CP05-bound M110-HW01 through M110-HW05 run with route, reset, staged soundcheck, rollback, SHA-bound listening and clean teardown evidence
- Operator accept or reject the two valid 2026-07-13 read-only preflight candidates; this decision-ledger requirement grants no live authority and does not reopen closed M005 behavior
- Exact WING Rack serial and same-console owner attestation for the immutable 2026-07-11 M005 evidence; this is identity-only and grants no live authority
- Clean committed M005 current-release compatibility replay plus an explicit verified_migration M006 owner decision bound to the recomputed migration and packet-verifier paths

Next operator action from `STATE.json`: Do not widen live authority. Connect the physical WING USB audio interface to this Mac and require one CoreAudio device named WING with 48 input and 48 output channels at 48 kHz; Dante Virtual Soundcard is not an accepted substitute. Provide the exact WING Rack serial and confirm it is the same console used for the 2026-07-11 M005 run, then build and reverify the M005 migration packet and record an explicit verified_migration M006 owner decision. After the dynamic CP05 gate passes the current non-marker tree, run M110-HW01-HW05 for route, reset, staged soundcheck, rollback, SHA-bound listening and clean teardown. Complete signing/notarization/Gatekeeper and production remote-auth recovery separately.

Latest ledger handoff row: `m111_m005_migration_fail_closed_handoff` at `2026-07-14T19:25:55Z`.

Historical 2026-07-12 ledger checkpoint: `local_engineering_95_final_tree_full_suite_no_wing_validation` at `2026-07-12T13:56:06Z`.

Live authority remains denied: release-ready live correction families remain `0`, live-allowed safety rows remain `0`, and current supervised live-write kinds remain `0`.

## Historical blocker trace — superseded

- M001 remains waiting for external validation, but the real WING Rack read-only
  artifact operator decision gate is no longer the active blocker. The current
  decision report is `accepted_decision_ready` with 1 accepted, 2 rejected, 0
  invalid and 0 unreviewed records.
- Supervised WING fader and gain-down proofs have now passed on the real
  console. The earlier failed rollback artifact
  `artifacts/wing_operator_proposal_validation/20260702T210956Z_wing_operator_proposal_ch3_fader/summary.json`
  (status `failed`, reason `rollback_not_applied`) is closed by the passed
  rerun `20260704T045741Z_wing_operator_proposal_ch3_fader` and the passed
  gain-down proof `20260704T045820Z_wing_operator_proposal_ch3_gain`; both
  carry apply, physical readback, rollback and emergency-stop/block evidence.
  The release readiness gate still requires the M005 disconnect/timeout/stale-state
  proof and the M006 owner readiness review.
- Release-ready live correction families remain `0` in
  `Docs/reports/WING_CORRECTION_APPLY_READINESS_20260703.md`; fader and gain
  are supervised policy candidates only, not release-ready live families.
- `backend/safety_coverage_matrix.py` has no live-allowed safety rows for this
  state. The fader and gain safety rows are now `hardware_proven_narrow` with
  passed apply/rollback/emergency-stop evidence, but `live_allowed` stays
  `false` for both until the M006 owner readiness review accepts the proof
  artifacts, so downstream autonomy promotion and mode-transition gates keep
  supervised live apply blocked.
- Operator product-state autonomy overlays also fail closed for this state:
  `supervised_live_allowed_static` is normalized to `false`, and the no-WING
  release gate rejects any overlay row that tries to claim it before proof
  passes.
- no-WING autonomy gap-rollup validation also fails closed for this state:
  rollup current live-write kinds/surfaces must match the mode-transition gate,
  so policy-level fader/gain cannot be accepted as current live authority while
  the gate exposes empty current live-write lists. The helper API now exposes
  this distinction as `policy_supervised_live_write_*` versus
  `mode_gate_supervised_live_write_*`, and the validator rejects the legacy
  gap-rollup claim phrase that called fader/gain the current supervised
  live-write boundary.
- Read-only preflight evidence also fails closed for readiness updates:
  `readiness_matrix_update_candidate_available` may be true for a clean
  candidate, but evidence-only `readiness_matrix_update_allowed` remains false.
  The same `readiness_matrix_update_allowed=false` denial is included in the
  read-only evidence `source_authority_flags`, and no-WING summary helpers
  re-check nested flags plus require an explicit source-authority-denied boolean
  before preserving readiness candidate availability; a separate manual
  review/update action is still required.
- Supporting real WING read-only evidence exists in
  `Docs/reports/WING_REAL_HARDWARE_READONLY_20260702.md`, but it is a
  temporary-root capture only. It cannot close readiness or authorize
  supervised writes.
- The next operator step is the M005 supervised disconnect/timeout/stale-state
  protection proof with the operator present, followed by the M006 owner
  readiness review that updates the MVP readiness matrix from the accepted
  CH3 fader and gain-down proof artifacts.
- A report-only M006 owner readiness review packet now exists at
  `artifacts/wing_owner_readiness_review/20260704T191101Z_m006_owner_readiness_review_packet/summary.json`
  and `Docs/reports/WING_M006_OWNER_READINESS_REVIEW_PACKET_20260704.md`. It
  prepares the owner review from the accepted CH3 fader/gain-down artifacts but
  is not an owner decision: M005 is still missing, readiness matrix updates are
  still disallowed, and live authority remains denied.
- `cca9fb447` adds `tools/verify_m006_owner_readiness_decision.py` and a
  pending owner decision template at
  `artifacts/wing_owner_readiness_review/20260704T191101Z_m006_owner_readiness_review_packet/owner_decision_template.json`.
  The decision verifier blocks the current template with
  `reason=m006_owner_decision_pending`; an accepted decision can only become a
  readiness-matrix update candidate for exactly `fader_supervised` and
  `gain_supervised` after a ready packet-verifier report and accepted M005
  summary path are supplied. This is offline verification only and grants no
  backend/WING contact, proposal creation, live apply, readiness update or live
  authority.
- `0936030c5` tightens that owner-decision gate: the ready packet-verifier
  report must include the current M005 stale-readback event-evidence checks
  before any accepted owner decision can become a readiness-matrix candidate.
  This prevents M006 from relying on stale packet-verifier reports created
  before `events.jsonl` stale-payload binding; it is offline verification only
  and does not close M005 or M006.
- `46f809911` tightens that owner-decision gate again: the ready
  packet-verifier report must also include the current M005 reconnect-readback
  event-evidence checks before any accepted owner decision can become a
  readiness-matrix candidate. This prevents M006 from relying on stale
  packet-verifier reports created before `events.jsonl` reconnect-readback
  binding; it is offline verification only and does not close M005 or M006.
- `5eda78126` binds that owner-decision gate to the exact M005 event-log path:
  the ready packet-verifier report must carry `m005_events` equal to the
  accepted M005 summary sibling `events.jsonl`. This prevents M006 from relying
  on missing, copied or mismatched event logs and still does not close M005 or
  M006.
Latest M005 readback target binding guard: `d3eb93a62`; M005 baseline, stale-readback and reconnect-readback payloads, including `events.jsonl` stale/reconnect payloads, now must match the requested `channel`, `value_type` and WING `address` before M005 completion or M006 owner-review input can be ready. This grants no backend/WING contact, proposal creation, live apply, readiness-matrix update or live authority.

- Latest M005 physical reconnect target guard: `bbf7833e2` hardens the M005
  runner before the next operator-present proof:
  executed `physical-cable-loss` mode now requires `--reconnect-wing-ip` before
  backend/WebSocket contact. This prevents a cable-loss proof attempt from
  producing a partial disconnected-write artifact without reconnect recovery
  and still does not close M005 or M006.
- `3ea749d70` tightens the packet-verifier side again: the M005
  reconnect-recovery physical readback summary must match a post-reconnect
  `events.jsonl` `recv` payload before owner-review input can be ready. This
  prevents hand-edited reconnect summary claims from satisfying M006 and still
  does not close M005 or M006.
- The M005/M006 verifier contract is now stricter: `f10894198` requires clean
  M005 owner-review input to be rollback-free. The M006 packet contract
  declares `required_rollback_request_sent=false` and forbids
  `rollback_last_manual_write` as a M005 remediation request type; the verifier
  rejects M005 summaries/events that contain rollback remediation. The previous
  `455a4028c` nested `operator_proposal_apply_blocked` response check,
  `b605b20f4` request/event sequence ordering, `223b8d433` canonical-path
  audit, `6cf8c8800` event-log audit, `f0ccd9842` request audit and
  `ad7f0da18` reconnect physical-readback checks remain in force. This is
  offline verification only; it does not close M005 or M006.
- Local runtime/tooling verification on 2026-07-04 is green but does not close
  the blocker: the current backend listener completed the 600.005 second
  read-only observation artifact
  `artifacts/soundcheck_runtime_observation/20260704T180231Z_soundcheck_runtime_observation/summary.json`
  with 2491 read-only meter frames, 119 dashboard snapshots, 0 live-write
  meter frames, 0 unexpected live events, no forbidden sent request types and
  no live apply request sent. Frontend test/build passed, targeted backend
  meter/runtime-observation/M005-runner/proposal tests passed, and WING MVP
  rails passed. `f8eb799c6` now requires future runtime observation summaries
  to include a complete `operator_runtime_contract` covering operator mode,
  connection state, dashboard, channel inventory, decision queue, safety
  monitor, meter subscription, read-only channel-level meters, analyzer status
  and master-bus status; the existing 600.005 second artifact recomputes as
  complete with no missing surfaces. Full backend validation also passed:
  `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`: 5482
  passed, 1 skipped. The no-WING quick gate failed closed on
  `backend_port_free` because the existing listener still needs operator
  confirmation. M005 still requires the operator-present physical cable-loss,
  stale-readback, blocked-write and reconnect proof, and M006 still requires
  owner readiness review.
- Offline long-run drift audit is now available, and it rejects the old daemon
  evidence instead of treating it as readiness proof. Audit artifact
  `artifacts/soundcheck_long_run_drift_audit/20260704T183743Z_soundcheck_long_run_drift_audit/summary.json`
  over `reports/soundcheck_daemon_log_20260703T221457.jsonl` failed because
  the source covered only 188.9 seconds, contained 287 live executor actions,
  73 positive gain/trim boosts, 422 relative-only fader/trim/gain targets and
  28 cumulative relative-drift surfaces. This is an offline negative check
  only: `live_authority_granted=false`, `readiness_closure_allowed=false` and
  `readiness_matrix_update_allowed=false`.
- Decision queue proposals now expose a read-only operator review coverage
  summary for `reason`, `risk`, `requested_change`, `old_value`, `new_value`
  and `rollback`. This makes missing proposal context visible to the operator
  without granting apply authority: it does not mutate the queue, does not run
  preflight, does not call `apply_operator_proposal`, and keeps
  `live_authority_granted=false`.
- Frontend task safety evidence now renders the same proposal review context
  before generic safety evidence: old value, new value, delta, rollback and
  review-field completeness. This is display-only operator visibility and does
  not grant apply authority, readiness closure or live WING authority.
- Dirty worktree split was recorded on `2026-07-04T18:27:09Z`: HEAD is
  `22ece58e09`, `ops/autopilot/WRITER_LEASE/owner.json` is absent, and
  `ops/autopilot/STOP` is absent. Keep WING proof/readiness WIP
  (`tools/wing_timeout_disconnect_proof_runner.py`, its tests/ADR and the
  active plan), soundcheck runtime WIP (meter stream backend, runtime
  observation runner/tests/artifact and frontend meter subscription), and
  frontend UI restyle WIP (`AutomixerDashboard.tsx`,
  `AutomixerComponents.tsx`, `DirectorPanel.tsx` and
  `Docs/reports/screenshots_frontend_restyle_20260704.png`) as separate commit
  groups. The UI restyle/screenshot must not be mixed into a WING readiness
  proof commit. The meter stream subscription remains read-only and now
  unsubscribes during frontend hook cleanup when the WebSocket is still open;
  frontend test/build passed after that cleanup. The reproducible split
  handoff is now recorded in
  `artifacts/worktree_split/20260704T185842Z_worktree_split/summary.json` and
  `Docs/reports/WORKTREE_SPLIT_20260704.md`. Generated review patches now live
  under `artifacts/worktree_split/20260704T185842Z_worktree_split/patches/`
  for WING proof/readiness WIP, soundcheck backend/runtime WIP, frontend
  runtime binding and frontend UI restyle. The patch bundle is not fully
  self-contained: the 60,032,994-byte runtime observation `events.jsonl`
  remains local evidence tracked by SHA-256 only, and the earlier
  `20260704T183538Z`/`20260704T183627Z` drift-audit summaries are superseded
  by the accepted `20260704T183743Z` audit summary. This split evidence grants
  no live authority and does not close M005 or M006. Pre-landing review patches
  01-04 and 06 passed `git apply --check` on a temporary clean detached
  worktree at `22ece58e0`; post-landing truth patch 05 is validated against
  the current landed split head. JSON checks, `git diff --check` and WING MVP
  rails also passed.
- The split has now been landed as separate local commits: `7a792457e` for
  read-only soundcheck runtime evidence, `3cf9b53a1` for frontend read-only
  meter binding, `e745438ab` for frontend UI restyle, `3f5d5c0eb` for the M005
  disconnect proof runner, and `9ce9e29ba` for the report-only M006 owner
  readiness packet. The large runtime observation `events.jsonl` and two
  superseded drift-audit attempts are ignored as local evidence only.
- Next operator action from `STATE.json`: Do not widen live authority. Treat M005 disconnect/timeout/stale-state behavior as closed and the narrow M006 readiness review owner decision as recorded; do not repeat the physical cable pull unless the owner explicitly requests it. For public release, rebind or migrate the closed M005 evidence to a clean current-source provenance packet, rerun the M006 technical verifier, reproduce the package from a clean checkout, complete Developer ID signing/notarization/Gatekeeper checks, and perform operator-present room/PA audible rollback validation.
- Latest ledger handoff row:
  `local_engineering_95_final_tree_full_suite_no_wing_validation` at
  `2026-07-12T13:56:06Z`. That row preserves the same `current_blocker`,
  `next_operator_action` and `external_validation_requirements` as
  `STATE.json`, so the safe handoff can be audited after mutable state moves.
- Live authority remains denied: release-ready live correction families remain
  `0`, live-allowed safety rows remain `0`, and current supervised live-write
  kinds remain `0`.
- `Docs/plans/M001-real-wing-interaction-plan.md` is synchronized to this
  blocker state: it shows `accepted_decision_ready`, 1 accepted, 2 rejected,
  0 invalid and 0 unreviewed decision records; the earlier
  `rollback_not_applied` supervised-proof blocker is closed by the passed
  CH3 fader and gain-down proofs. M005 behavior and M006 owner intent are now
  closed; current-source promotion remains fail-closed.
- `Docs/reports/WING_READONLY_PREFLIGHT_HANDOFF_PACKET_LATEST.md` is scoped as
  read-only preflight evidence handoff only. It no longer stands alone as the
  current next-action source after `accepted_decision_ready`; current
  operator action must come from `Docs/CURRENT_STATE.md`, the active
  real-WING plan and `ops/autopilot/STATE.json`.
- The current worktree contains local WING evidence/report updates and SmartDSP
  scoring work. Future autopilot runs must avoid overwriting unrelated local
  changes if they appear.
- Future heartbeat wake-ups must verify native Goal transition tools before
  replacing a thread Goal.

## Historical external validation requirements — superseded

- Exact WING Rack serial and same-console owner attestation for the immutable 2026-07-11 M005 evidence; this is identity-only and grants no live authority
- Clean committed M005 current-release compatibility replay plus an explicit verified_migration M006 owner decision bound to the recomputed migration and packet-verifier paths

## Not blockers

- `ops/autopilot/STOP` does not exist after setup.
- No active writer lease is present; `ops/autopilot/WRITER_LEASE/owner.json`
  is absent and the runtime lease directory is ignored.
- Control-plane docs/state can proceed without live hardware contact.
