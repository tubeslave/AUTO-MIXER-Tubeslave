# Roadmap

Status: active dependency-ordered roadmap

Roadmap выбирается от текущего evidence state. Milestone считается валидным
только если он finite, independently verifiable и имеет rollback.

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

## Active milestone

Exactly one milestone is active: `M001`, currently
`WAITING_FOR_EXTERNAL_VALIDATION`. See `Docs/ACTIVE_MILESTONE.md` and
`Docs/plans/M001-wing-readonly-preflight-evidence-gate.md`.
