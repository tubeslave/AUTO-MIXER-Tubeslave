# Autopilot Heartbeat

Last heartbeat UTC: 2026-07-03T12:11:35Z

Status: WAITING_FOR_EXTERNAL_VALIDATION

Active milestone: M001

Active checkpoint: M001-EXTERNAL-WING-READONLY-ARTIFACT

Writer lease: none; `ops/autopilot/WRITER_LEASE/owner.json` is absent and
`ops/autopilot/WRITER_LEASE/` is ignored as runtime state.

Next action: do not widen live authority. The decision report is now
`accepted_decision_ready`, but the CH3 fader proof failed rollback with reason
`rollback_not_applied`. Operator must restore or verify CH3 fader state, then
rerun the narrow fader proof until apply, physical readback, rollback and
emergency-stop/block evidence all pass. Gain proof must be separate and
headroom-safe.

External validation requirements:

- Operator restores or verifies CH3 fader state after rollback_not_applied
- Clean supervised CH3 fader proof rerun with apply, physical readback, rollback and emergency-stop/block evidence
- Separate headroom-safe supervised gain-down proof with the same evidence requirements

Latest check: the control-plane state, safety coverage matrix and operator
product-state autonomy overlay are synchronized to the current WING
decision/apply-readiness reports. `Docs/plans/M001-real-wing-interaction-plan.md`
now also mirrors the same accepted decision ledger and failed CH3 rollback
blocker instead of the stale `operator_review_required` gate wording.
`Docs/ACTIVE_MILESTONE.md`, `Docs/ROADMAP.md` and the M001 plan now also state
that the open external requirement is no longer read-only artifact review; it is
CH3 fader restoration/verification after `rollback_not_applied`, a clean
supervised fader proof rerun, and then a separate headroom-safe gain-down proof.
`Docs/reports/WING_READONLY_PREFLIGHT_HANDOFF_PACKET_LATEST.md` now declares
that it is read-only preflight evidence handoff only, and that current
next-action authority after `accepted_decision_ready` comes from current state
and the active real-WING plan, not from the old handoff packet by itself.
Latest verified full pytest passed with `4821 passed, 1 skipped, 1 warning`.
Repo-local no-WING quick passed with failed_checks `[]` and no live authority at
`artifacts/no_wing_release_check/20260703T101148Z_no_wing_release_check/summary.json`.
The current no-WING evidence rail now also reloads
`artifact-manifest.json`, recomputes `summary.json` and `summary.md` SHA256
values, runs `verify_no_wing_artifact_manifest`, and checks manifest
self-consistency plus cross-surface denial fields. This catches stale or
tampered current evidence instead of trusting the state string alone.
The latest ledger authority rail now also binds the newest `RUN_LEDGER.jsonl`
entry to the same fail-closed state while M001 is waiting for external
validation: `hardware_contacted=false`, `live_writes=false`,
`readiness_matrix_update_allowed=false`, release-ready live families `0`, live
allowed safety rows `0`, current supervised live-write kinds `0`, and no
unrestricted live/soundcheck authority.
The operator-report full-suite rail now also binds this heartbeat and
`LAST_REPORT.md` to the same latest full-suite result recorded in `STATE.json`
and the newest full-suite `RUN_LEDGER.jsonl` row, so the operator-facing green
test claim cannot stay stale after a new validation pass.
`STATE.json` now also exposes the same safe handoff text as
`next_operator_action`, and the WING MVP rails require it to stay synchronized
with `next_action` while the external rollback/gain proof blocker is open.
The latest ledger handoff rail now also requires the newest `RUN_LEDGER.jsonl`
row to record the same `current_blocker`, `next_operator_action` and
`external_validation_requirements` as `STATE.json`, so the audit trail keeps the
safe handoff even after mutable state moves forward.
The BLOCKERS handoff rail now also requires `ops/autopilot/BLOCKERS.md` to
record the same external validation requirements, `STATE.json`
`next_operator_action`, and latest ledger handoff row. This keeps the primary
blocker file from drifting away from machine state while M001 is externally
blocked.
The active milestone and M001 plan handoff rail now also requires
`Docs/ACTIVE_MILESTONE.md` and
`Docs/plans/M001-wing-readonly-preflight-evidence-gate.md` to preserve the same
external validation requirements, `STATE.json` next operator action and latest
ledger handoff row, so milestone-level docs cannot drift into a vague rollback
summary.
The roadmap handoff rail now also requires `Docs/ROADMAP.md` to preserve the
same external validation requirements, `STATE.json` next operator action and
latest ledger handoff row, so the top-level dependency roadmap cannot drift
from machine state while M001 is externally blocked.
The project-layer blocker handoff rail now also requires
`.ai/project-layer/wing_mvp_blockers.md` to preserve the same exact handoff
without changing its first-row blocker status selection.
The MVP progress handoff rail now also requires
`Docs/reports/MVP_PROGRESS_LATEST.md` to preserve the same exact handoff,
latest ledger row and strict `94/100` pre-release score gate, so the
progress truth source cannot stay on a stale snapshot or imply `95/100` /
public-release readiness while rollback and gain proof remain open.
That artifact reports read-only preflight candidate availability as separate
from permission: `readiness_matrix_update_candidate_available=true`, while
`readiness_matrix_update_allowed=false`, `readiness_closure_allowed=false` and
`live_authority_granted=false`.
Its handoff verifier summary surface also matches with mismatches `[]`,
`readiness_matrix_update_allowed=false`, `readiness_closure_allowed=false` and
`live_authority_granted=false`.
Its read-only evidence `source_authority_flags` also includes
`readiness_matrix_update_allowed=false`, so downstream consumers cannot treat
the authority-denial packet as complete while omitting readiness-matrix update
permission.
no-WING summary helpers also re-check the nested authority flags before
surfacing `source_authority_denied=true`, and they now require that reported
boolean to be explicitly present.
Supporting real WING read-only tmp capture was recorded in
`Docs/reports/WING_REAL_HARDWARE_READONLY_20260702.md`: channel 3 produced 15
readback values through the existing backend session, with no proposal request,
no write request, no live authority and no readiness closure. It is not a
canonical readiness candidate because its artifact root is under `/tmp`, not
`artifacts/wing_readonly_preflight`.
Latest blocker truth sync: `ops/autopilot/BLOCKERS.md` now reflects that the
canonical read-only decision ledger is accepted-ready with 1 accepted, 2
rejected, 0 invalid and 0 unreviewed records, while release readiness remains
blocked by failed CH3 fader rollback proof and zero release-ready live
correction families. The safety coverage matrix now has
`live_allowed_row_ids: []`; `fader_supervised` is `supervised_candidate` with
`rollback_readback: failed` and `emergency_stop_proof: missing`. Product-state
autonomy overlays now also normalize `supervised_live_allowed_static` to
`false`, and the no-WING release gate rejects rows that try to set it true.
no-WING gap-rollup validation now also requires
`supervised_current_live_write_kinds` and surfaces to match the mode-transition
gate, and the helper API now names that split explicitly:
`policy_supervised_live_write_*` is policy fader/gain, while
`mode_gate_supervised_live_write_*` is the current empty gate authority. The
gap-rollup validator also rejects the old claim phrase that treated fader/gain
as the current supervised live-write boundary.
Read-only preflight evidence now also separates candidate availability from
permission: clean candidates can surface for manual review, but evidence-only
`readiness_matrix_update_allowed` stays `false`.
The same readiness-matrix denial is now part of the read-only evidence
`source_authority_flags` contract, and no-WING summary helpers re-check nested
flags before preserving an explicitly reported source-authority-denied boolean.
WING MVP blocker rails passed with `125 passed`, affected autonomy/registry
tests passed with `61 passed`, focused read-only readiness split checks passed
with `121 passed`, readiness authority-flag focused checks passed with
`15 passed`, no-WING source-authority summary regression checks passed with
`4 passed`, explicit-denial regression checks passed with `4 passed`, affected
read-only/no-WING/proof-plan tests passed with `656 passed`, no-WING
release-check tests passed with `455 passed`, the latest
targeted operator-overlay checks passed with
`7 passed`, the latest targeted gap-rollup/claim checks passed with `5 passed`,
the targeted STOP/lease/recovery rail passed with `1 passed`, and Project
Director routing tests passed with `9 passed`.
Current targeted surface-contract checks passed:
`tests/test_supervised_wing_proof_plan.py` `186 passed`, selected WING MVP
rails `4 passed`, `py_compile` passed, JSON/JSONL validation passed, and
`tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_no_wing_surface_contract_20260702T0815Z`
passed.
Current SmartDSP weighted-score and WING-native fit evidence checks passed:
selected strategy-search/operator-explanation checks `2 passed`, WING-native
fit tie-break checks `2 passed`, `tests/test_smart_dsp_strategy_engine.py`
`249 passed`, `py_compile` passed, and full suite `4796 passed, 1 skipped, 1 warning`.
SmartDSP now exposes mode-specific weighted score contributions and
WING-native strategy-fit summaries/gaps for selected and alternative
strategies in offline/dry-run evidence only. The fit selector may prefer a
close-score WING Rack fw3.0.5 native OSC strategy only with no blockers and no
lower safety score; no WING live authority, readiness closure, queue import, or
live write scope changed.
Current SmartDSP DSP-parameter calculation summary checks passed:
selected muddy vocal soundcheck strategy `1 passed`,
`tests/test_smart_dsp_strategy_engine.py` `249 passed`, selected WING MVP rails
`120 passed`, and `py_compile` passed. SmartDSP now exposes
`dsp_parameter_calculation_summary` for the selected strategy and alternatives,
including calculated DSP parameter keys/values, calculation source fields,
WING OSC apply/review/readback keys, WING-native fit status, and
readback/rollback requirements. This remains proposal-only/offline evidence:
`proposal_only=true`, `live_write=false`, no readiness closure and no new WING
EQ/dynamics/FX/pan/master authority.
Current SmartDSP problem-first action decision checks passed:
muddy vocal and vocal masking level-plane tests `2 passed`,
`tests/test_smart_dsp_strategy_engine.py` `249 passed`, and `py_compile`
passed. SmartDSP now exposes `problem_action_decision_summary` /
`selected_problem_action_decision_summary` for selected strategies and
alternatives, including selected action intent, considered method families,
considered action intents, method-family score leaders, top alternative actions
and `selection_basis`. This makes balance-before-EQ, static-EQ-vs-dynamic-EQ,
de-esser-vs-static-EQ and other strategy choices auditable as
problem-first decisions, not fixed-method cards. It remains proposal-only:
`proposal_only=true`, `live_write=false`, no readiness closure and no new WING
EQ/dynamics/FX/pan/master authority.
Current runner artifact surface-contract checks passed:
`tests/test_wing_operator_proposal_validation_runner.py` `21 passed`, and
`py_compile tools/wing_operator_proposal_validation_runner.py` passed.
Current smart-DSP gate-slot conflict checks passed:
`tests/test_smart_dsp_strategy_engine.py` `244 passed`,
`tests/test_smart_dsp_replay_evaluation.py` `92 passed`, and
`py_compile backend/smart_dsp_replay_evaluation.py backend/smart_dsp_strategy_engine.py`
passed.
Current SmartDSP WING readback evidence extraction checks passed:
`tests/test_smart_dsp_evidence_extractor.py` `83 passed`, combined
SmartDSP strategy/replay checks `339 passed`, and extractor/strategy
`py_compile` passed.
Current WING proof apply surface-contract checks passed:
selected `tests/test_server.py` apply/preflight mismatch and linked rollback
coverage `5 passed`, selected product-state preflight/plan/dry-run checks
`3 passed`, handler/queue `py_compile` passed, full suite passed, and
`tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_no_wing_basic_controls_20260702T0850Z`
passed.


Supporting real WING read-only evidence was also captured into a temporary root:
`/tmp/automixer_real_wing_verified_subset_20260702T_tmp_persist/wing_readonly_preflight/20260702T094851Z_wing_readonly_preflight/summary.json`. It passed with 15 physical readbacks and no writes, but it
does not close readiness because the canonical review contract only accepts
readiness candidates from `artifacts/wing_readonly_preflight`. See
`Docs/reports/WING_REAL_HARDWARE_READONLY_20260702.md`.

The latest real WING read-only candidate remains
`candidate_ready_for_manual_review`, and the decision report is now
`accepted_decision_ready` for the canonical `20260702T095758Z` source id. This
does not grant live authority or release readiness; the next blocker is the
failed CH3 fader rollback proof and the lack of any release-ready live
correction family.

This heartbeat includes one local product/runtime change: SmartDSP strategy
search now reports mode-specific weighted score contributions, WING-native
strategy-fit evidence, DSP-parameter calculation summaries, and problem-first
action decision summaries for selected and alternative strategies in
offline/dry-run evidence. No WING/backend contact was made by this state-sync
cycle, and no live console authority changed. The
committed `20260702T075151Z` artifact records read-only backend/WING
gate-parameter queries with no live write, and the latest proof-plan runtime
commit binds future fader/gain proof steps to concrete WING surfaces and
negative-only directions. The operator runbook now repeats the same
surface/address contract, and the runner now writes those addresses into
dry-run/physical summaries and proposals. The smart-DSP replay suite now covers
the fail-closed WING gate-slot model conflict path, and the evidence extractor
can build SmartDSP current-state inputs from already captured read-only WING
preflight artifacts. The operator proposal apply path now rejects mismatched
WING proof surfaces/addresses/directions before any manual console write call.
No OSC write path was opened by this cycle. No live write, readiness closure,
live authority grant, accepted hardware evidence or Full Autonomous readiness
was produced. WING EQ/dynamics readbacks are imported as read-only state but do
not count as verified processor-model proof. WING fader, input-trim and pan
readbacks are also imported as read-only SmartDSP analysis context only; `pan`
does not become a current WING deployment apply target. WING gain-up candidates
now also consult the read-only current fader value and block when the projected
fader would cross the 0 dB ceiling. If WING control readback context exists but
the current fader value is missing, gain-up is also blocked until the fader
readback is available. SmartDSP operator proposals now also include an
evidence-extraction contract that lists required evidence components and their
coverage without granting live authority. The apply path now blocks positive
WING proof-direction moves even for operator-authored proof proposals, and
read-only evidence intake ignores validate-only/no-transport preflight traces
so they do not pollute the real candidate queue. The real WING Rack is treated
as connected and available for supervised tests, but permission is limited to
necessary fader/gain-down proof actions after accepted read-only decisions; it
does not grant unrestricted EQ, dynamics, FX, routing, master, scene, live or
soundcheck authority.
