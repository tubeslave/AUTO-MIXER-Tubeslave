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

## Active milestone

Exactly one milestone is active: `M001`, currently
`WAITING_FOR_EXTERNAL_VALIDATION`. See `Docs/ACTIVE_MILESTONE.md` and
`Docs/plans/M001-wing-readonly-preflight-evidence-gate.md`.
