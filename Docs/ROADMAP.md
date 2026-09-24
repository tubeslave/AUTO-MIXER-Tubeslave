# Roadmap

Status: active dependency-ordered roadmap

Roadmap выбирается от текущего evidence state. Milestone считается валидным
только если он finite, independently verifiable и имеет rollback.

## M120 route-teardown overlay — 2026-07-15

The non-silent source and serial-bound WING USB route are now machine-proven by
accepted artifact `4937d06623084a2d98e7bc3f9b8f8a17`; the earlier M116
digital-silence checkpoint remains historical. M120 repairs only the runner's
disconnect-teardown classification and does not widen live authority. The next
finite capability step is a fresh clean committed M120/CP05-bound HW01 repeat
with verified teardown, followed by HW02-HW05 only after physical output
containment. Strict readiness stays `88/100` until the staged real-WING proof
and other external gates close.

## M118 sealed migration-replay overlay — 2026-07-15

M118 preserves the exact-zero no-network replay contract while preventing a
known urllib3 import-time IPv6 probe under active guards. Project modules still
observe the real IPv6 capability. The old source-bound challenge is invalid;
the finite next sequence is source commit, dynamic CP05 rebind, fresh challenge,
owner same-console attestation and a private-root sealed replay. This adds no
live authority and does not replace M110 HW01-HW05.

## M117 current-truth overlay — 2026-07-15

M117 closes a named control-plane gap: operator-attended diagnostic contact is
kept distinct from live authority, and non-machine-verified capture observations
are labelled as such on every current handoff surface. It adds no processing,
hardware or live-write capability. Strict readiness remains `88/100` and the
next capability-changing step is still the attended M110 HW01-HW05 sequence.

## M110 operator-directed overlay — 2026-07-14

The frozen M001-M010 dependency table below is not rewritten. Current active
work is the separate M110 attended path recorded in `STATE.json.active_work`:
CP00-CP05 complete under the source-bound marker contract. The clean M115
checkout reached WING Rack serial `0100Q1V0604AAE`, firmware
`3.0.5-4-ga4ea833d:release`, and opened CoreAudio `WING` at 48x48/48 kHz.
The machine-recorded route gate then failed closed before artifact issuance on
`master_channel_signal_missing`. Operator-observed short and 120-second
diagnostics showed exact digital zero across all 48 channels; those extra
observations are not independently machine-verified. HW01 is incomplete and
HW02-HW05 remain pending; reset, corrections and writes remain zero.

M115 records the committed `cc1a2ca50` validation baseline (`7214 passed, 3
skipped`, 136 rails, 66/66 no-WING) and preserves the M113 CoreAudio
fail-closed identity gate: both collection and independent evidence verification
now require exact integer 48x48/48 kHz metadata. The next finite sequence now
starts only when a confirmed non-silent source reaches Main 47/48; then in one
operator-attended process prove route, mute/contain outputs, bypass the 64
CH1-16 processor-enable addresses, restore only the 77 known M006 changed
addresses through exact 23/14/14 stages, unmute, analyze, execute interleaved
processor/musical lanes, perform the final audit, confirm the exact state SHA by
listening, repeat the audit, and verify teardown. Any CAS, headroom, connection,
freshness, writer, rollback or readiness failure stops the run.

M005 is behaviorally closed and M006 permission is recorded for that one path;
neither creates standing authority. Strict real-live readiness remains
`88/100` until HW05 proves `show_mix_ready`; normal allowlist remains `0` and
the `95/100` target is not yet achieved.

## Current full-project gate — 2026-07-12

Local P1 overlay completed: the soundcheck parent is decomposed into semantic
children; local blockers no longer poison siblings; bounded replans and the
only currently trusted constructive substitute (complete down-only fader
translation) are content-bound and approval-gated; final readiness is
`safe_core_ready`, `show_mix_ready` or `soundcheck_incomplete` from fresh
physical/audio evidence. Kernel score is `96/100`; strict real-live soundcheck
is `88/100` because standing authority is zero and the current staged flow has
not yet completed on the real console.

The local reanalysis control plane is complete: candidate/commit/discard,
default-deny observation shadow, immutable accepted-bundle identity and
derivation+monitor+console-writer fencing are adversarially verified. This
closes a local architectural blocker but grants no physical authority.

Strict readiness is `88/100`, not `95/100`; public release ready is false and
standing supervised live authority is empty. Local arm64 packaging and
source-bound no-WING runtime smoke are verified, but the package is ad-hoc
signed, dirty-tree-bound and `releaseEligible=false`.

The next finite external sequence is intentionally narrow:

1. commit M120, rebind CP05 once, repeat HW01 with the proven non-silent
   WING/Main 47/48 source, then perform the attended staged soundcheck proof;
2. record accepted or rejected decisions for the two valid 2026-07-13
   read-only candidates without inheriting authority from older evidence;
3. rebind the already closed M005 behavioral artifact to current-source
   provenance without another cable pull unless the owner explicitly asks,
   then rerun the M006 technical verifier without widening any unproven family;
4. separately complete clean-checkout reproduction, Developer ID/notary/
   Gatekeeper, production remote-auth recovery and room/PA validation.

## Current external validation requirements

- Repeat M110-HW01 from the clean committed M120 source after one marker-only CP05 rebind; keep the already proven non-silent Main 47/48 feed active and require a fresh accepted route artifact plus verified capture stop, mixer disconnect, route revocation and clean backend termination
- Operator-present CP05-bound M110-HW01 through M110-HW05 run with route, reset, staged soundcheck, rollback, SHA-bound listening and clean teardown evidence
- Operator accept or reject the two valid 2026-07-13 read-only preflight candidates; this decision-ledger requirement grants no live authority and does not reopen closed M005 behavior
- Exact WING Rack serial and same-console owner attestation for the immutable 2026-07-11 M005 evidence; this is identity-only and grants no live authority
- Clean committed M005 current-release compatibility replay plus an explicit verified_migration M006 owner decision bound to the recomputed migration and packet-verifier paths

No live authority changed: release-ready live correction families remain `0`,
live-allowed safety rows remain `0`, and current supervised live-write kinds
remain `0`.

Current blocker from `STATE.json`: M110-HW01 is no longer blocked by digital silence. On clean CP05-bound source 45e2a2ae66853dc79a8db693f8ccfe15cbc67cc8, accepted route artifact /Users/dmitrijvolkov/.automixer/runtime/m110_master_route_proofs/4937d06623084a2d98e7bc3f9b8f8a17.json verified the exact WING serial and firmware, serial-bound USB/CoreAudio 48x48/48 kHz identity, two complete Main 47/48 MAIN L/R route passes, fresh signal at conservative -12.318 dBTP, and 0 console mutation writes. The runner remained blocked only by proof_owned_master_route_teardown_failed:wing_disconnect_failed:RuntimeError because the expected post-disconnect readiness_invalidated broadcast was treated as fatal; the backend log shows capture stop, route revocation, WING/OSC disconnect and clean backend shutdown. M120 adds a teardown-only exact-envelope receiver; independent rereview is ACCEPT and 142 runner tests passed. The pre-fix attempt cannot close HW01: a fresh clean committed M120/CP05-bound route-only repetition with verified teardown is required. HW02-HW05 remain pending; reset, corrections and live writes remain 0. M111 still requires exact same-console attestation, a clean committed compatibility replay and an explicit migration-bound M006 decision. Strict real-live readiness stays 88/100: normal allowlist 0, standing authority false, safe_core_ready/show_mix_ready unproven.

Next operator action from `STATE.json`: Do not widen live authority. Finish M120 validation, commit the source/truth change, release the writer lease, create exactly one marker-only CP05 rebind and verify the dynamic gate. Keep the proven non-silent Ableton-to-WING Main 47/48 feed running and repeat route-only M110-HW01 from a clean committed checkout. Proceed to HW02-HW05 only if the fresh artifact is accepted and teardown verifies capture stopped, mixer disconnected, route authority revoked and backend terminated cleanly. Before any HW02-HW05 write, physically contain all PA and monitor outputs and confirm Main 1 muted. Separately provide the exact WING Rack serial/same-console attestation for the 2026-07-11 M005 run, complete the clean committed compatibility replay and record an explicit verified_migration M006 owner decision. Complete signing/notarization/Gatekeeper and production remote-auth recovery separately.

Latest ledger handoff row: `offline_phase_verification` at `2026-09-24T08:04:58Z`.

The detailed table below is a frozen dependency taxonomy, not the operational
task selector; its old M002-M006 status labels are historical. Current release
truth and the complete requirements before `95/100` are owned by
`ops/autopilot/STATE.json.strict_pre_release_review`.

## Selection Rules

Следующий milestone выбирается по порядку:

1. safety and architectural prerequisites;
2. dependency readiness;
3. measurable capability progress;
4. reduction of major technical uncertainty;
5. contribution to end-to-end Automixer operation;
6. project risk;
7. expected engineering value.

Documentation cleanup, cosmetic refactoring и speculative architecture work не
выбираются, если есть dependency-ready capability milestone.

## Milestones

| ID | Title | Desired end state | Dependencies | Safety class | Target maturity transition | Status |
| --- | --- | --- | --- | --- | --- | --- |
| M001 | WING read-only preflight evidence gate | No-WING prestart is green, run inputs are validated, and the next real WING read-only artifact path is unambiguous | Existing WING preflight tooling | high, read-only | SPECIFIED to REPLAY_VERIFIED for gate tooling; hardware remains WAITING | WAITING_FOR_EXTERNAL_VALIDATION |
| M002 | Physical console readback intake | Real read-only WING artifact is reviewed and bound to accepted decision records | M001 plus WING hardware/operator | critical, read-only hardware | HARDWARE_READ_VERIFIED for readback rows | WAITING_FOR_EXTERNAL_VALIDATION |
| M003 | Supervised fader proof | Exact bounded fader move is approved, applied, read back and rolled back | M002 | critical live write | SUPERVISED_LIVE_VERIFIED for fader only | BLOCKED_BY_M002 |
| M004 | Supervised gain-down/trim proof | Exact bounded gain-down/trim move is approved, applied, read back and rolled back | M002 | critical live write | SUPERVISED_LIVE_VERIFIED for gain-down only | BLOCKED_BY_M002 |
| M005 | Emergency stop and timeout proof | Emergency stop, disconnect, timeout and stale-state protections are verified | M002 | critical safety | SIMULATOR_VERIFIED plus supervised hardware evidence | BLOCKED_BY_M002 |
| M006 | Operator evidence and readiness update | MVP readiness matrix updates only from accepted real proof | M002-M005 | high audit | HARDWARE_READ_VERIFIED or SUPERVISED_LIVE_VERIFIED evidence surfaced | BLOCKED_BY_M002 |
| M007 | Offline capability promotion framework | Long-term processing families have replay/fault/evaluation gates before any live expansion | M001-M006 | high offline | PROTOTYPE/UNIT_VERIFIED to REPLAY_VERIFIED | PENDING |
| M008 | Assisted processing family expansion | EQ, dynamics, pan, FX and balance stay proposal-only with source identity and safety gates | M007 | high assisted | UNIT_VERIFIED to INTEGRATION_VERIFIED | PENDING |
| M009 | Supervised non-fader/gain live expansion | A separately approved ADR widens supervised live writes one family at a time | M006-M008 plus hardware | critical live write | SUPERVISED_LIVE_VERIFIED per family | PENDING_APPROVAL |
| M010 | Full autonomous eligibility review | Full Autonomous Mixing Mode eligibility is reviewed against all completion gates | M001-M009 | critical autonomous | AUTONOMOUS_ELIGIBLE only if all gates pass | PENDING_APPROVAL |
| M011 | Offline replay evaluation contract | Soundcheck, studio and concert replay manifests fail closed until approved thresholds and complete scenario evidence exist | M007-M010 | high offline | SPECIFIED to UNIT_VERIFIED for gate contract; replay evidence remains pending | PENDING |
| M012 | Fault-injection recovery contract | Simulator fault bundles fail closed until required failure modes, recovery outcomes and audit evidence exist | M011 | high offline | SPECIFIED to UNIT_VERIFIED for recovery gate; simulator evidence remains pending | PENDING |
| M013 | Integration evaluation bundle contract | Replay and fault-recovery manifests are composed into an offline integration bundle before promotion | M011-M012 | high offline | SPECIFIED to UNIT_VERIFIED for integration gate; integration evidence remains pending | PENDING |
| M014 | Quality threshold policy contract | Candidate quality thresholds are complete, evidence-bound and owner-approved before replay or integration promotion | M011-M013 | high offline | SPECIFIED to UNIT_VERIFIED for threshold gate; owner approval remains required | PENDING |
| M015 | Replay threshold policy binding | Offline replay promotion consumes the quality threshold policy contract instead of loose threshold status fields | M011-M014 | high offline | UNIT_VERIFIED gate binding; replay evidence remains pending | PENDING |
| M016 | Recovery policy contract | Candidate recovery policies are complete, evidence-bound and owner-approved before simulator fault-recovery promotion | M012-M015 | high offline | SPECIFIED to UNIT_VERIFIED for recovery-policy gate; owner approval remains required | PENDING |
| M017 | Fault recovery policy binding | Simulator fault-recovery promotion consumes the recovery policy contract instead of loose recovery_policy status fields | M012-M016 | high offline | UNIT_VERIFIED gate binding; simulator evidence remains pending | PENDING |
| M018 | Integration policy contract | Candidate integration policies are complete, evidence-bound and owner-approved before offline integration promotion | M013-M017 | high offline | SPECIFIED to UNIT_VERIFIED for integration-policy gate; owner approval remains required | PENDING |
| M019 | Integration policy binding | Offline integration promotion consumes the integration policy contract instead of loose integration_policy status fields | M013-M018 | high offline | UNIT_VERIFIED gate binding; integration evidence remains pending | PENDING |
| M020 | Safety kernel trace contract | Deterministic safety-kernel traces are complete, audited and non-bypassable before integration use | M013-M019 | high offline | SPECIFIED to UNIT_VERIFIED for safety-kernel trace gate; integration binding remains pending | PENDING |
| M021 | Safety kernel trace binding | Offline integration promotion consumes the safety-kernel trace contract instead of loose safety_kernel_trace_bound checks | M013-M020 | high offline | UNIT_VERIFIED gate binding; integration evidence remains pending | PENDING |
| M022 | No-live-transport contract | Offline integration candidates prove no live transport, backend contact, console contact or live write occurred | M013-M021 | high offline | SPECIFIED to UNIT_VERIFIED for no-live-transport gate; integration binding remains pending | PENDING |
| M023 | No-live-transport binding | Offline integration promotion consumes the no-live-transport contract instead of loose no_live_transport checks | M013-M022 | high offline | UNIT_VERIFIED gate binding; integration evidence remains pending | PENDING |
| M024 | Operator report contract | Offline integration reports prove operator-visible status, blockers, authority denial, child evidence and next action before review use | M013-M023 | high offline | SPECIFIED to UNIT_VERIFIED for operator-report gate; integration binding remains pending | PENDING |
| M025 | Operator report binding | Offline integration promotion consumes the operator-report contract instead of loose operator_report_bound checks | M013-M024 | high offline | UNIT_VERIFIED gate binding; integration evidence remains pending | PENDING |
| M026 | Autonomy readiness integration bundle overlay | Autonomy promotion readiness consumes offline integration bundle contract status before planning supervised or autonomous promotion | M007-M025 | high offline | UNIT_VERIFIED readiness overlay; live authority remains blocked | PENDING |
| M027 | Operator adapter integration bundle readiness summary | Operator-facing module status rows surface offline integration bundle readiness, blockers and authority-denial status from autonomy promotion readiness | M026 | high offline frontend | UNIT_VERIFIED read-only operator surface; live authority remains blocked | PENDING |
| M028 | Main pipeline preview integration bundle readiness summary | Main pipeline preview summary surfaces offline integration bundle readiness, blockers and authority-denial status from autonomy promotion readiness | M026-M027 | high offline preview | UNIT_VERIFIED read-only preview summary; live authority remains blocked | PENDING |
| M029 | Director runtime integration bundle readiness summary | Director runtime surfaces main pipeline preview integration bundle readiness, blockers and authority-denial status for the runtime guard | M027-M028 | high offline frontend | UNIT_VERIFIED read-only Director surface; live authority remains blocked | PENDING |
| M030 | Director event integration bundle blocker summary | Director event rows render integration bundle blocker tags as operator-readable read-only events | M029 | high offline frontend | UNIT_VERIFIED read-only Director event surface; live authority remains blocked | PENDING |
| M031 | Director guard integration bundle blocker banner | Runtime Guard blocker banner renders integration bundle blocker tags as operator-readable read-only messages | M030 | high offline frontend | UNIT_VERIFIED read-only Director guard banner; live authority remains blocked | PENDING |
| M032 | Director goal integration bundle blocker summary | Director current goal names the offline integration bundle blocker before generic promotion-planning fallback | M031 | high offline frontend | UNIT_VERIFIED read-only Director goal; live authority remains blocked | PENDING |
| M033 | Director queue integration bundle blocker badge | Director proposal queue badge shows blocked promotion planning when the offline integration bundle is blocked | M032 | high offline frontend | UNIT_VERIFIED read-only Director queue badge; live authority remains blocked | PENDING |
| M034 | Director suggestions integration bundle blocker notice | Agent Suggestions panel shows a read-only integration bundle blocker notice while suggestions remain review-only | M033 | high offline frontend | UNIT_VERIFIED read-only Director suggestions notice; live authority remains blocked | PENDING |
| M035 | Task Queue integration bundle blocker notice | Task Queue shows a read-only integration bundle blocker notice before proposal actions when promotion planning is blocked | M034 | high offline frontend | UNIT_VERIFIED read-only Task Queue notice; live authority remains blocked | PENDING |
| M036 | Supervised Apply integration bundle blocker notice | Supervised Apply modal shows a read-only integration bundle blocker notice before supervised apply review | M035 | high offline frontend | UNIT_VERIFIED read-only Supervised Apply notice; live authority remains blocked | PENDING |
| M037 | Supervised Apply integration bundle blocker enforcement | Supervised Apply modal and task actions block apply confirm when promotion planning is blocked | M036 | high offline frontend | UNIT_VERIFIED read-only Supervised Apply enforcement; live authority remains blocked | PENDING |
| M038 | Supervised Apply confirm callback blocker guard | Supervised Apply confirm callback returns before dispatch while promotion planning is blocked | M037 | high offline frontend | UNIT_VERIFIED guarded confirm callback; live authority remains blocked | PENDING |
| M039 | Task apply request helper blocker guard | Task apply request helper returns before opening Supervised Apply while promotion planning is blocked | M038 | high offline frontend | UNIT_VERIFIED guarded task apply request; live authority remains blocked | PENDING |
| M040 | Task Details apply entrypoint blocker guard | Task Details apply button stays blocked before opening Supervised Apply while promotion planning is blocked | M039 | high offline frontend | UNIT_VERIFIED guarded task details apply entrypoint; live authority remains blocked | PENDING |
| M041 | Task Queue metrics blocker guard | Task Queue metrics show blocked apply capacity while promotion planning is blocked | M040 | high offline frontend | UNIT_VERIFIED blocked Task Queue metrics; live authority remains blocked | PENDING |
| M042 | Task Queue review filter blocker guard | Task Queue Review only filter includes all rows while promotion planning is blocked | M041 | high offline frontend | UNIT_VERIFIED blocked Task Queue review filter; live authority remains blocked | PENDING |
| M043 | Task Queue row badge blocker guard | Task Queue rows show blocked badges while promotion planning is blocked | M042 | high offline frontend | UNIT_VERIFIED blocked Task Queue row badges; live authority remains blocked | PENDING |
| M044 | Task Details badge blocker guard | Task Details status badges show blocked state while promotion planning is blocked | M043 | high offline frontend | UNIT_VERIFIED blocked Task Details badges; live authority remains blocked | PENDING |
| M045 | Task Queue review-only metric blocker guard | Task Queue Review Only metric counts all rows while promotion planning is blocked | M044 | high offline frontend | UNIT_VERIFIED blocked Task Queue review-only metric; live authority remains blocked | PENDING |
| M046 | Task Details blocker list blocker guard | Task Details Apply Blockers list shows integration bundle blockers while promotion planning is blocked | M045 | high offline frontend | UNIT_VERIFIED blocked Task Details blocker list; live authority remains blocked | PENDING |
| M047 | Apply Plan sequence blocker guard | Apply Plan and modal Gate Sequence show blocked integration bundle steps while promotion planning is blocked | M046 | high offline frontend | UNIT_VERIFIED blocked Apply Plan sequence; live authority remains blocked | PENDING |
| M048 | Operator Decision blocker status guard | Operator Decision panel shows integration bundle blocker status while promotion planning is blocked | M047 | high offline frontend | UNIT_VERIFIED blocked Operator Decision status; live authority remains blocked | PENDING |
| M049 | Supervised Apply readiness blocker guard | Supervised Apply modal Final readiness shows integration bundle blocker status while promotion planning is blocked | M048 | high offline frontend | UNIT_VERIFIED blocked Supervised Apply readiness; live authority remains blocked | PENDING |
| M050 | Supervised Apply backend result blocker guard | Supervised Apply modal hides matched backend results while promotion planning is blocked | M049 | high offline frontend | UNIT_VERIFIED hidden Supervised Apply backend result; live authority remains blocked | PENDING |
| M051 | Supervised Apply gate row blocker guard | Supervised Apply modal Apply Gate rows show no-contact and no-live status while promotion planning is blocked | M050 | high offline frontend | UNIT_VERIFIED blocked Supervised Apply gate rows; live authority remains blocked | PENDING |
| M052 | Task Details apply button blocker guard | Task Details apply button shows blocked state while promotion planning is blocked | M051 | high offline frontend | UNIT_VERIFIED blocked Task Details apply button; live authority remains blocked | PENDING |
| M053 | Task Queue row reject blocker guard | Task Queue row actions hide reject while promotion planning is blocked | M052 | high offline frontend | UNIT_VERIFIED blocked Task Queue row reject action; live authority remains blocked | PENDING |
| M054 | Task Details reject button blocker guard | Task Details reject button shows blocked state while promotion planning is blocked | M053 | high offline frontend | UNIT_VERIFIED blocked Task Details reject button; live authority remains blocked | PENDING |
| M055 | Operator Decision apply result blocker guard | Operator Decision Apply result row shows blocked state while promotion planning is blocked | M054 | high offline frontend | UNIT_VERIFIED blocked Operator Decision apply result; live authority remains blocked | PENDING |
| M056 | Safety Monitor policy blocker guard | Safety Monitor Policy row shows blocked state while promotion planning is blocked | M055 | high offline frontend | UNIT_VERIFIED blocked Safety Monitor policy; live authority remains blocked | PENDING |
| M057 | Safety Monitor goal readiness blocker guard | Safety Monitor Goal readiness row shows blocked state while promotion planning is blocked | M056 | high offline frontend | UNIT_VERIFIED blocked Safety Monitor goal readiness; live authority remains blocked | PENDING |
| M058 | Safety Monitor blocked surfaces blocker guard | Safety Monitor Blocked footer shows integration bundle blocker while promotion planning is blocked | M057 | high offline frontend | UNIT_VERIFIED blocked Safety Monitor blocked surfaces; live authority remains blocked | PENDING |
| M059 | Replay Evidence decision trace status blocker guard | Replay Evidence Trace row shows integration bundle blocker status while promotion planning is blocked | M058 | high offline frontend | UNIT_VERIFIED blocked Replay Evidence decision trace status; live authority remains blocked | PENDING |
| M060 | Replay Evidence latest run status blocker guard | Replay Evidence Latest run badge shows integration bundle blocker status while promotion planning is blocked | M059 | high offline frontend | UNIT_VERIFIED blocked Replay Evidence latest run status; live authority remains blocked | PENDING |
| M061 | Replay Evidence policy row blocker guard | Replay Evidence Policy row shows integration bundle blocker status while promotion planning is blocked | M060 | high offline frontend | UNIT_VERIFIED blocked Replay Evidence policy row; live authority remains blocked | PENDING |
| M062 | Replay Evidence channel status blocker guard | Replay Evidence channel status badge shows integration bundle blocker status while promotion planning is blocked | M061 | high offline frontend | UNIT_VERIFIED blocked Replay Evidence channel status; live authority remains blocked | PENDING |
| M063 | Replay Evidence report path blocker guard | Replay Evidence report path text shows integration bundle blocker status while promotion planning is blocked | M062 | high offline frontend | UNIT_VERIFIED blocked Replay Evidence report path; live authority remains blocked | PENDING |
| M064 | Task Queue status column blocker guard | Task Queue Status column shows integration bundle blocker status while promotion planning is blocked | M063 | high offline frontend | UNIT_VERIFIED blocked Task Queue status column; live authority remains blocked | PENDING |
| M065 | Task Details status value blocker guard | Task Details primary status badge shows integration bundle blocker status while promotion planning is blocked | M064 | high offline frontend | UNIT_VERIFIED blocked Task Details status value; live authority remains blocked | PENDING |
| M066 | Task Details proposal risk blocker guard | Task Details Proposal Preview risk badge shows integration bundle blocker status while promotion planning is blocked | M065 | high offline frontend | UNIT_VERIFIED blocked Task Details proposal risk; live authority remains blocked | PENDING |
| M067 | Task Queue move delta blocker guard | Task Queue Target / Move delta shows integration bundle blocker status while promotion planning is blocked | M066 | high offline frontend | UNIT_VERIFIED blocked Task Queue move delta; live authority remains blocked | PENDING |
| M068 | Task Details proposal delta blocker guard | Task Details Proposal Preview delta shows integration bundle blocker status while promotion planning is blocked | M067 | high offline frontend | UNIT_VERIFIED blocked Task Details proposal delta; live authority remains blocked | PENDING |
| M069 | Supervised Apply new value blocker guard | Supervised Apply Proposal Target New value shows integration bundle blocker status while promotion planning is blocked | M068 | high offline frontend | UNIT_VERIFIED blocked Supervised Apply new value; live authority remains blocked | PENDING |
| M070 | Supervised Apply rollback blocker guard | Supervised Apply Proposal Target Rollback shows no-live-apply status while promotion planning is blocked | M069 | high offline frontend | UNIT_VERIFIED blocked Supervised Apply rollback; live authority remains blocked | PENDING |
| M071 | Task Details proposal rollback blocker guard | Task Details Proposal Preview Rollback shows no-live-apply status while promotion planning is blocked | M070 | high offline frontend | UNIT_VERIFIED blocked Task Details proposal rollback; live authority remains blocked | PENDING |
| M072 | Task Details proposal proof blocker guard | Task Details Proposal Preview Proof shows no-live-proof status while promotion planning is blocked | M071 | high offline frontend | UNIT_VERIFIED blocked Task Details proposal proof; live authority remains blocked | PENDING |
| M073 | Task Details proposal parameter blocker guard | Task Details Proposal Preview Parameter shows blocked-parameter status while promotion planning is blocked | M072 | high offline frontend | UNIT_VERIFIED blocked Task Details proposal parameter; live authority remains blocked | PENDING |
| M074 | Supervised Apply parameter blocker guard | Supervised Apply Proposal Target Parameter shows blocked-parameter status while promotion planning is blocked | M073 | high offline frontend | UNIT_VERIFIED blocked Supervised Apply parameter; live authority remains blocked | PENDING |
| M075 | Supervised Apply old value blocker guard | Supervised Apply Proposal Target Old value shows no-readback status while promotion planning is blocked | M074 | high offline frontend | UNIT_VERIFIED blocked Supervised Apply old value; live authority remains blocked | PENDING |
| M076 | Replay Evidence start action blocker guard | Replay Evidence Start Offline action is disabled while promotion planning is blocked | M075 | high offline frontend | UNIT_VERIFIED blocked Replay Evidence start action; live authority remains blocked | PENDING |
| M077 | Replay Evidence latest run id blocker guard | Replay Evidence Latest run id shows no-replay-start status while promotion planning is blocked | M076 | high offline frontend | UNIT_VERIFIED blocked Replay Evidence latest run id; live authority remains blocked | PENDING |
| M078 | Replay Evidence trace detail blocker guard | Replay Evidence Trace detail shows no-trace-proof status while promotion planning is blocked | M077 | high offline frontend | UNIT_VERIFIED blocked Replay Evidence trace detail; live authority remains blocked | PENDING |
| M079 | Replay Evidence channel identity blocker guard | Replay Evidence channel identity shows no-channel-proof status while promotion planning is blocked | M078 | high offline frontend | UNIT_VERIFIED blocked Replay Evidence channel identity; live authority remains blocked | PENDING |
| M080 | Replay Evidence report row blocker visibility guard | Replay Evidence report row remains visible with blocker status while promotion planning is blocked | M079 | high offline frontend | UNIT_VERIFIED visible blocked Replay Evidence report row; live authority remains blocked | PENDING |
| M081 | WING review decision prestart proof binding | Review decision templates and ledgers bind accepted WING candidates to backend prestart and manifest-verifier proof fields | M002 | critical read-only hardware review | UNIT_VERIFIED stricter WING review decision proof binding; hardware remains waiting | WAITING_FOR_EXTERNAL_VALIDATION |
| M082 | WING handoff decision template proof fields | Operator handoff packets surface the review-decision template required fields and prestart/verifier proof-field subset | M002 | critical read-only hardware handoff | UNIT_VERIFIED handoff proof-field contract; hardware remains waiting | WAITING_FOR_EXTERNAL_VALIDATION |
| M083 | Source identity and arrangement read-only context gates | Source-identity and arrangement-role proposal candidates require fresh read-only/no-authority context before downstream mix decisions can trust them | M008 | high offline analysis | UNIT_VERIFIED read-only context gate; live authority remains blocked | PENDING |
| M084 | Unmasking, frequency and balance context gates | Cross-channel unmasking, frequency separation and bus/group balance proposal candidates require fresh read-only/no-authority context before downstream mix decisions can trust them | M083 | high offline mix decision | UNIT_VERIFIED downstream context gate; live authority remains blocked | PENDING |
| M085 | FX send monitor and return safety gates | FX send proposal candidates require fresh read-only/no-authority send/return context, mapped no-authority FX schema hints, monitor-safety evidence and rollback/readback sources | M083 | high offline FX decision | UNIT_VERIFIED FX send safety gate; live authority remains blocked | PENDING |
| M086 | Master bus PA feed and reference safety gates | Master-bus proposal candidates require fresh read-only/no-authority main-bus and PA-feed context, explicit PA-feed safety and reference-feed exclusion | M084 | critical offline master output | UNIT_VERIFIED master/output safety gate; live authority remains blocked | PENDING |
| M087 | Performance adaptation read-only context gates | Performance-adaptation proposal candidates require fresh read-only/no-authority current performance context before concert-time level adaptation decisions can trust them | M083 | high offline adaptive mixing | UNIT_VERIFIED performance-adaptation context gate; live authority remains blocked | PENDING |
| M088 | Routing management read-only context gates | Routing proposal candidates require fresh read-only/no-authority current route context before console-routing decisions can trust them | M083 | critical offline console routing | UNIT_VERIFIED routing context gate; live authority remains blocked | PENDING |
| M089 | Mute management read-only context gates | Mute/unmute proposal candidates require fresh read-only/no-authority current mute context before console-mute decisions can trust them | M083 | critical offline console mute | UNIT_VERIFIED mute context gate; live authority remains blocked | PENDING |
| M090 | Pan spatial read-only context gates | Pan proposal candidates require fresh read-only/no-authority current pan and mono/phase context before pan/stereo decisions can trust them | M083 | critical offline console pan | UNIT_VERIFIED pan context gate; live authority remains blocked | PENDING |
| M091 | Scene/preset management read-only context gates | Scene/preset proposal candidates require fresh read-only/no-authority scene/preset context, scoped diffs and rollback evidence before destructive scene decisions can trust them | M083 | critical offline console scene/preset | UNIT_VERIFIED scene/preset context gate; live authority remains blocked | PENDING |
| M092 | Mix variant profile read-only context gates | Mix-variant proposal candidates require fresh read-only/no-authority genre/task context, trusted source identity, reference metrics and arrangement context before genre/task variants can fan out | M083 | high offline genre/task mix decision | UNIT_VERIFIED mix-variant context gate; live authority remains blocked | PENDING |
| M093 | Decision evidence packet read-only context gates | Decision-evidence packet candidates require fresh read-only/no-authority decision context, trusted source identity, readback state and rollback planning before evidence packets can rank downstream proposals | M083 | high offline decision basis | UNIT_VERIFIED decision-evidence context gate; live authority remains blocked | PENDING |
| M094 | Project Director external-validation scope guard | Project Director narrows autonomous work to WING external-validation handoff/intake or control-plane state while M001 waits for real WING/operator evidence | M001 | critical autopilot direction | UNIT_VERIFIED external-validation scope guard; live authority remains blocked | WAITING_FOR_EXTERNAL_VALIDATION |
| M095 | WING read-only reserved-root intake guard | Real WING read-only intake rejects spoofed candidates from reserved no-WING and autopilot evidence roots before manual review can accept them | M001 | critical read-only hardware intake | UNIT_VERIFIED reserved-root intake guard; live authority remains blocked | WAITING_FOR_EXTERNAL_VALIDATION |
| M096 | WING read-only declared path intake guard | Real WING read-only intake rejects candidates whose summary/events paths do not match declared artifact_root and run_id before manual review can accept them | M001 | critical read-only hardware intake | UNIT_VERIFIED declared artifact path-shape guard; live authority remains blocked | WAITING_FOR_EXTERNAL_VALIDATION |
| M097 | WING read-only canonical root intake guard | Real WING read-only intake rejects candidates whose artifact_root is not the canonical WING read-only capture root before manual review can accept them | M001 | critical read-only hardware intake | UNIT_VERIFIED canonical artifact-root guard; live authority remains blocked | WAITING_FOR_EXTERNAL_VALIDATION |
| M098 | WING read-only source path intake guard | Real WING read-only intake rejects copied or stale-source summary files whose source path does not match the declared summary_path before manual review can accept them | M001 | critical read-only hardware intake | UNIT_VERIFIED source-path binding guard; live authority remains blocked | WAITING_FOR_EXTERNAL_VALIDATION |
| M099 | WING read-only run id path segment intake guard | Real WING read-only intake rejects candidates whose run_id is not a single safe path segment before manual review can accept them | M001 | critical read-only hardware intake | UNIT_VERIFIED run-id path-segment guard; live authority remains blocked | WAITING_FOR_EXTERNAL_VALIDATION |
| M100 | WING read-only artifact root parent traversal intake guard | Real WING read-only intake rejects candidates whose artifact_root contains parent traversal before manual review can accept them | M001 | critical read-only hardware intake | UNIT_VERIFIED artifact-root traversal guard; live authority remains blocked | WAITING_FOR_EXTERNAL_VALIDATION |
| M101 | WING read-only artifact root whitespace intake guard | Real WING read-only intake rejects candidates whose artifact_root has surrounding whitespace before manual review can accept them | M001 | critical read-only hardware intake | UNIT_VERIFIED artifact-root whitespace guard; live authority remains blocked | WAITING_FOR_EXTERNAL_VALIDATION |
| M102 | WING read-only run id format intake guard | Real WING read-only intake rejects candidates whose run_id is not the runner-generated UTC capture format before manual review can accept them | M001 | critical read-only hardware intake | UNIT_VERIFIED run-id format guard; live authority remains blocked | WAITING_FOR_EXTERNAL_VALIDATION |
| M103 | WING read-only run id timestamp intake guard | Real WING read-only intake rejects candidates whose run_id timestamp is not a valid UTC capture time before manual review can accept them | M001 | critical read-only hardware intake | UNIT_VERIFIED run-id timestamp guard; live authority remains blocked | WAITING_FOR_EXTERNAL_VALIDATION |
| M104 | WING read-only capture time order intake guard | Real WING read-only intake rejects candidates whose captured_at is earlier than the UTC capture timestamp encoded in run_id | M001 | critical read-only hardware intake | UNIT_VERIFIED capture-time ordering guard; live authority remains blocked | WAITING_FOR_EXTERNAL_VALIDATION |
| M105 | Project Director post-M104 intake loop guard | Project Director detects the post-M104 WING read-only intake-hardening streak and narrows further autonomous work to real external validation or a named unverified contract gap | M001 | critical autopilot direction | UNIT_VERIFIED post-M104 intake-loop guard; live authority remains blocked | WAITING_FOR_EXTERNAL_VALIDATION |
| M106 | Project Director post-M105 intake loop persistence | Project Director keeps the post-M104 intake-loop guard active even after direction-guard commits sit above the WING read-only streak | M001 | critical autopilot direction | UNIT_VERIFIED post-M105 loop persistence; live authority remains blocked | WAITING_FOR_EXTERNAL_VALIDATION |
| M107 | Project Director intake loop lookback floor | Project Director keeps the post-M104 intake-loop guard active even when additional direction-guard commits would otherwise push the WING read-only streak outside the requested lookback window | M001 | critical autopilot direction | UNIT_VERIFIED intake-loop lookback floor; live authority remains blocked | WAITING_FOR_EXTERNAL_VALIDATION |
| M123 | Instrument Space decision integration | Measured scene relations and drift deterministically select existing analyzers and annotate proposal review without creating console authority | M122 read-only slice; owner-directed parallel software work | high, advisory-only | INTEGRATION_VERIFIED software evidence; attended WING/dLive proof remains separate | WAITING_FOR_EXTERNAL_VALIDATION |

## Active milestone

Exactly one milestone is active: `M001`, currently
`WAITING_FOR_EXTERNAL_VALIDATION`. See `Docs/ACTIVE_MILESTONE.md` and
`Docs/plans/M001-wing-readonly-preflight-evidence-gate.md`.

Current M001 waiting reason: the aggregate WING read-only decision ledger is
`operator_review_required` because two valid 2026-07-13 candidates remain
unreviewed. The historical exact source
`artifacts/wing_readonly_preflight/20260702T095758Z_wing_readonly_preflight/summary.json`
retains its `accepted_decision_ready` record, but later candidates inherit
neither that decision nor live authority. The supervised CH3 fader/gain-down
proofs passed with apply, physical readback, rollback and emergency-stop/block
evidence, and the earlier `rollback_not_applied` artifact remains closed audit
history. M005 disconnect/timeout/stale-state behavior is closed and the narrow
M006 owner decision is recorded; no roadmap row advances to standing live
authority until the pending decisions, current-source provenance, M110 hardware
proof and technical promotion pass.

Latest M005 readback target binding guard: `d3eb93a62`; M005 baseline, stale-readback and reconnect-readback payloads, including `events.jsonl` stale/reconnect payloads, now must match the requested `channel`, `value_type` and WING `address` before M005 completion or M006 owner-review input can be ready. This grants no backend/WING contact, proposal creation, live apply, readiness-matrix update or live authority.

Latest M005 physical reconnect target guard: `bbf7833e2`; executed `physical-cable-loss` M005 proofs now fail closed before backend/WebSocket contact unless `--reconnect-wing-ip` is supplied. This keeps the operator-present cable-loss proof bound to reconnect-recovery evidence and grants no backend/WING contact, proposal creation, live apply, readiness-matrix update or live authority.

Latest M006 packet verifier hardening: `3ea749d70`; it requires a ready
packet-verifier report with the current M005 stale-readback event-evidence
checks and matching reconnect-recovery physical readback event checks from
`events.jsonl`.

Latest M006 owner decision verifier: `5eda78126`; it requires a ready
packet-verifier report with the current M005 stale-readback event-evidence
checks, matching reconnect-recovery physical readback event checks and the
accepted M005 summary sibling `events.jsonl` path. The owner decision is
recorded, but technical promotion blocks with
`reason=m006_owner_decision_failed_closed` because current-source M005
provenance is not accepted; it does not grant a readiness-matrix update.
