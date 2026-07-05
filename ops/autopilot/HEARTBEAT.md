# Autopilot Heartbeat

Last heartbeat UTC: 2026-07-05T01:50:44Z

Status: WAITING_FOR_EXTERNAL_VALIDATION

Active milestone: M001

Active checkpoint: M001-EXTERNAL-WING-READONLY-ARTIFACT

Writer lease: none; `ops/autopilot/WRITER_LEASE/owner.json` is absent and
`ops/autopilot/WRITER_LEASE/` is ignored as runtime state.

Next action: do not widen live authority. The decision report is
`accepted_decision_ready`, and the CH3 fader/gain-down proofs have passed
with apply, physical readback, rollback and emergency-stop/block evidence.
The earlier `rollback_not_applied` artifact is closed by the rerun, but M005
physical disconnect/timeout/stale-state proof and M006 owner readiness review
remain external gates.

External validation requirements:

- Supervised WING disconnect/timeout/stale-state protection proof (M005) with operator present
- Owner readiness review that updates the MVP readiness matrix from the accepted CH3 fader and gain-down proof artifacts (M006)

Latest check: `46f809911` requires the offline M006 owner-decision verifier to reject ready packet-verifier summaries unless the current M005 reconnect-readback event checks are present and true alongside stale-readback checks. This heartbeat records offline verifier/test hardening only: no WING/backend contact during this checkpoint, no proposal creation, no live apply, no readiness-matrix update and no authority widening.
Latest verified full pytest passed with `5504 passed, 1 skipped, 1 warning`.
Latest direct mute/pan invalid-value guard passed: direct WebSocket
`set_mute`, `mute_channel` and `unmute_channel` now reject missing or
unparseable mute values before mixer writes, and direct `set_pan` rejects
malformed, boolean, NaN or infinite values instead of silently coercing them to
center pan; focused checks, server tests, WING rails, adjacent WING/client/FX
tests and full pytest passed.
Latest voice command numeric validation guard passed: malformed, boolean,
NaN or infinite numeric inputs for direct fader/gain/pan, volume relative dB,
EQ band gain and compressor threshold/gain now emit explicit
`voice_command_error` payloads with live/readiness authority denied before any
mixer readback or write method can run; focused checks, server tests, WING
rails, adjacent WING/client/FX/mixer tests and full pytest passed.
Latest voice command malformed payload guard passed: non-mapping payloads
such as raw strings or lists now emit explicit `voice_command_error` payloads
with `reason=malformed_voice_command`, `status=error`, `success=false` and
live/readiness authority denied before any mixer write/readback method can run;
focused checks, server tests, WING rails and full pytest passed.
Latest voice command invalid request guard passed: unknown command types, missing
`channel`, missing `snap_name`, missing EQ band, missing compressor threshold
and unsupported mixer-client paths now emit explicit `voice_command_error`
payloads with `status=error`, `success=false` and live/readiness authority
denied; invalid requests do not call mixer write/readback methods; focused
checks, server tests, WING rails and full pytest passed.
Latest voice command error authority guard passed: no-mixer and exception
paths now emit explicit `voice_command_error` payloads with `status=error`,
`success=false`, surface/request/channel metadata and live/readiness authority
denied instead of silent no-op or underspecified errors; `CancelledError` still
re-raises; focused checks, server tests, WING rails and full pytest passed.
Latest voice command write-result guard passed: direct voice `set_fader`,
`set_gain`, `set_pan`, `mute_channel`, `volume_up`/`volume_down`, EQ and
compressor write commands now emit fail-closed `voice_command_blocked`
responses when mixer writes return `False`; cached fader state is not advanced
after failed volume voice writes; focused checks, server tests and full pytest
passed with no WING/backend contact or live authority changes.
Latest supervised rollback retry runbook guard passed: manual-write retry delay now waits two full throttle intervals, proof-sequence regression covers apply/rollback/emergency-stop/block, and the runbook uses the current accepted read-only source id with write-attempt guidance; targeted server checks passed and the full suite stayed green.
Latest WING client write-result guard passed: `set_channel_mute` and `set_channel_pan` now return the underlying `send` result, so dropped writes can fail closed in callers; targeted WING client checks passed and the full suite stayed green.
External broad live soundcheck v2 reports observed: `reports/live_wing_soundcheck_v2_original_20260703T211128.json` and `reports/live_wing_soundcheck_v2_applied_20260703T211128.json` are not release proof and require operator review/rollback decision before any readiness claim.
External broad live soundcheck v2 stage2 reports observed: `reports/live_wing_soundcheck_v2s2_original_20260703T212521.json` and `reports/live_wing_soundcheck_v2s2_applied_20260703T212521.json` are not release proof and require operator review/rollback decision before any readiness claim.
External broad live soundcheck v2 stage3 reports observed: `reports/live_wing_soundcheck_v2s3_original_20260703T213559.json` and `reports/live_wing_soundcheck_v2s3_applied_20260703T213559.json` are not release proof and require operator review/rollback decision before any readiness claim.
Latest live soundcheck static acknowledgement contract passed: `tests/test_write_surface_inventory_static_scan.py` now enumerates `tools/live_wing_soundcheck*.py`, requires the exact acknowledgement before config/client/audio-device context, and confirms blocked write-surface inventory coverage; targeted checks passed and the full suite stayed green.
Latest raw OSC send inventory guard passed: direct `send(..., value)` patterns are now treated as write-adjacent by the static scan; meter subscription transport, enhanced gain direct apply, legacy snapshot navigation scripts and `tools/live_wing_soundcheck_daemon.py` are inventoried as non-authoritative blocked/quarantined surfaces; targeted checks and full pytest passed.
Latest WING client bridge write-result guard passed: MixerClientBase-compatible fader/gain/mute/pan/HPF/polarity/delay/send-level methods now return underlying `send()` results, and low-cut/delay/send multi-write helpers fail closed if any sub-write fails; targeted WING client checks and full pytest passed.
Latest WING client all-write-result guard passed: all direct `WingClient.set_*` wrappers that call `send()` now return the underlying or aggregate `send()` result, including EQ, dynamics, gate, bus, main, matrix, FX, insert, reverb and channel-send helpers; targeted WING client checks, static inventory checks, WING MVP rails and full pytest passed.
Latest mixer mute/pan write-result guard passed: direct WebSocket `set_mute`, `mute_channel`, `unmute_channel` and `set_pan` paths now report blocked/error when mixer writes return `False` or raise, instead of silently swallowing failed mute/pan writes; targeted checks, server tests, WING MVP rails and full pytest passed.
Latest FX handler write-result guard passed: FX/insert direct-write WebSocket handlers now return blocked `mixer_write_failed` or `mixer_write_exception` responses when mixer writes fail, instead of emitting `insert_updated`, `fx_slot_updated` or `fx_parameter_updated`; targeted checks, server tests, WING MVP rails and full pytest passed.
Latest AutoFX unsupported action guard passed: `AutoFXPlanner` no longer emits `set_fx_on` in blocked FX apply evidence; WING FX slot activity is documented as loaded-model-derived, the unsupported `/fx/N/on` action is excluded from `blocked_calls`, AutoFX remains proposal-only, and focused checks, WING MVP rails and full pytest passed.
Latest WING FX unsupported path guard passed: `WingClient.set_fx_on` now fails closed instead of writing nonexistent `/fx/N/on`; FX slot read/query helpers use official FW 3.1 model/fxmix metadata, `set_reverb_params` avoids the unsupported path, observation-mode dry runs do not shadow it, and stage3 rollback snapshots skip it; targeted checks, WING MVP rails and full pytest passed.
Latest auth token digest storage guard passed: `backend/auth.py` now stores HMAC-SHA256 digests of issued bearer tokens instead of raw token strings in `AuthManager._tokens`, while generate/validate/revoke/cleanup behavior stays intact; targeted auth checks passed and the full suite stayed green.
Latest auth token owner rate-limit guard passed: `AuthManager.authenticate` now rejects caller `client_id` mismatches and rate-limits by the issued token owner even when `client_id` is omitted; targeted auth checks passed and the full suite stayed green.
Latest broad live WING soundcheck setup guard passed: `tools/live_wing_soundcheck_setup.py` remains blocked from non-dry-run execution without `--confirm-live-wing-write I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING`, and `backend/write_surface_inventory.py` records it as a blocked write surface; targeted write-surface checks passed and the full suite stayed green.
Latest broad live WING soundcheck v2 guard passed: `tools/live_wing_soundcheck_v2.py` is now recorded as a blocked write surface, and non-dry-run execution requires `--confirm-live-wing-write I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING` before config or `WingClient` access; targeted write-surface checks passed and the full suite stayed green.
Latest broad live WING soundcheck v2 stage2 guard passed: `tools/live_wing_soundcheck_v2_stage2.py` is now recorded as a blocked write surface, and non-dry-run execution requires `--confirm-live-wing-write I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING` before config or `WingClient` access; targeted write-surface checks passed and the full suite stayed green.
Latest Pulse export secret redaction guard passed: ChatGPT Pulse snapshot excerpts now redact bearer tokens, `sk-`/`sk-proj-` keys, JSON-style `apiKey`/`accessToken` fields and assignment-style API key/token/password/secret values before JSON/Markdown export; targeted checks passed and the full suite stayed green.
Latest broad live WING correction test guard passed: `tools/live_wing_correction_test.py` now requires `--confirm-live-wing-write I_UNDERSTAND_THIS_WRITES_TO_LIVE_WING` before loading mixer config or creating a `WingClient`, and `backend/write_surface_inventory.py` records it as a blocked write surface; targeted write-surface checks passed and the full suite stayed green.
Latest vector index generic secret redaction guard passed: generic assignment-style secret values such as `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, `HF_TOKEN`, `DATABASE_PASSWORD`, `SLACK_BOT_TOKEN` and `CLIENT_SECRET` are redacted before indexing, while non-secret tokenizer assignments remain unchanged; targeted vector/Paperclip/operator checks passed and the full suite stayed green.
Latest vector index secret intake guard passed: project vector indexing now rejects secret-like config filenames such as `.env.local`, `secrets.json`, `api_keys.yaml` and `service-account.json`, and manual `vector-add --file` refuses secret-like file paths before reading; targeted vector/Paperclip/operator checks passed and the full suite stayed green.
Latest owned subprocess timeout sweep passed: operator, TrainerAgent, REPA offline-render, Ayaic MP3 and repo-local CLI test subprocess calls now carry explicit timeouts; TrainerAgent timeouts return a failed result instead of crashing; broad owned-code AST scan reports `NO_OWNED_SUBPROCESS_TIMEOUT_GAPS`; targeted checks and the full suite stayed green.
Latest selected tool subprocess timeout contract passed: FFmpeg export/decode helpers, latency git metadata and dataset hash subprocess calls now carry explicit timeouts; targeted/offline checks passed and the full suite stayed green.
Latest subprocess probe timeout guard passed: server lsof port inspection and telemetry git metadata probes now use bounded timeouts and fail closed on timeout; targeted server/telemetry checks passed and the full suite stayed green.
Latest C++ bridge failed-start cleanup guard passed: AutoFader native startup now terminates partial processes on readiness timeout or startup failure, with kill fallback if graceful termination hangs; targeted checks passed and the full suite stayed green.
Latest training dataset discovery path guard passed: Hugging Face file names are validated before URL construction or download path selection, unsafe metadata paths are skipped, targeted checks passed, and the full suite stayed green.
Latest agent-training archive guard passed: ZIP/TAR dataset archives now validate all member paths and reject traversal, absolute, Windows-root, link and special-type entries before selecting a local training file; targeted checks passed, archive extraction scan stayed clean, and the full suite stayed green.
Latest dLive TLS verification guard passed: TLS mode now verifies certificates and hostnames by default, targeted checks passed, and the full suite stayed green.
Latest torch checkpoint guard validation passed: MuQ-Eval and local ONNX export checkpoint loaders require `weights_only=True`, targeted checks passed, and the full suite stayed green.
Latest ML NPZ pickle guard validation passed: training dataset loaders use `allow_pickle=False`, targeted ML checks passed, and the full suite stayed green.
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
The structured pre-release score rail now also requires `STATE.json`
`strict_pre_release_review`, `Docs/reports/MVP_PROGRESS_LATEST.md`,
`Docs/CURRENT_STATE.md` and `LAST_REPORT.md` to agree on score `95/100`,
threshold `95/100`, `public_release_ready=false` and the rollback/gain
proof score ceiling. This is a report-only guard and grants no live
authority.
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
Latest blocker truth sync: `ops/autopilot/BLOCKERS.md` reflects that the
canonical read-only decision ledger is accepted-ready with 1 accepted, 2
rejected, 0 invalid and 0 unreviewed records, and that the earlier failed CH3
fader rollback proof is closed by the passed rerun. Release readiness remains
blocked by M005/M006 and zero release-ready live correction families. The
safety coverage matrix still has `live_allowed_row_ids: []`, and product-state
autonomy overlays normalize `supervised_live_allowed_static` to `false`.
Current marker: release readiness remains
blocked until M005 and M006 pass.
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
does not grant live authority or release readiness; the next blockers are M005
disconnect/timeout/stale-state proof, M006 owner readiness review and the lack
of any release-ready live correction family.

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

Latest M005 software disconnect protection evidence (partial): on the real WING, software disconnect made physical readback unavailable, an approved supervised write attempt produced no applied write, and reconnect restored verified readback with CH3 unchanged at 0.0. Physical cable-loss/timeout phases were skipped by operator decision and remain the open M005 evidence. See artifacts/wing_timeout_disconnect_proof/20260704T074537Z_wing_m005_disconnect_timeout_stale_proof/summary.json.
