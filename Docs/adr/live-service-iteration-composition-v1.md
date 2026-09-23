# Live service iteration composition v1

Date: 2026-09-23
Status: implemented, software evidence pending/CI-gated
Scope: `backend/live_runtime`

## Decision

`LiveSoundcheckService` is the canonical owner of the realtime one-hypothesis loop. Sequential `MixFeatures` snapshots now advance the live state through proposal, application, verification and either KEEP, verified rollback or HOLD.

The service composes the existing pieces instead of creating a second decision authority:

`MixFeatures/evidence -> Director -> IterationCoordinator -> LiveControlPlane -> WingWriteAdapter -> WING readback -> Critic -> KEEP | rollback | HOLD`

A second proposal is not evaluated while an iteration is waiting for verification. The verification window is non-blocking: later feature snapshots advance the Critic. A terminal iteration returns to LISTEN; operator takeover or failed restoration propagates HOLD.

## Evidence rules

`main_peak_dbfs` is taken directly from each `MixFeatures` snapshot. Source-harshness verification is taken from the target channel in the same snapshot. Metrics produced by other realtime analyzers, for example vocal intelligibility, may be supplied explicitly as `verification_metrics`.

Missing verification evidence is not synthesized from a legacy heuristic. The Critic fails closed, which sends a reversible action through verified rollback.

`operator_took_control` is carried into Critic evidence as `operator_touch`. Operator priority remains absolute: HOLD is entered without an automatic rollback that might overwrite the engineer's manual correction.

## Safety and mode contract

The service does not infer BENCH_TEST from mixer connectivity. The active `LiveStartRequest.mode` remains authoritative. Production safety policy, fresh read-before/write/readback and rollback verification remain below the iteration layer.

A WING transport replacement while a hypothesis is in flight is rejected rather than silently moving the experiment and rollback baseline to another transport instance.

## Renovation classification

- **KEEP_CORE**: WING/OSC transport, subscriptions/callback readback, typed contracts, safety/freeze policy, verified control-plane rollback.
- **ADAPT**: `LiveSoundcheckService`, `IterationCoordinator`, realtime feature/metric evidence, and the temporary `AutoSoundcheckEngine` bridge only for discovery/audio/physical connection plumbing.
- **ARCHIVE**: legacy AutoFOH pending-action orchestration, proxy evaluation/rollback construction, AutoEQ/AutoFader decision policy after imports/runtime references are severed and replacement HIL evidence exists.
- **DELETE_AFTER_PROOF**: none added by this change. No legacy module is deleted before reference severing plus replacement tests/HIL.

## Test gate

`tests/test_live_runtime_service_iteration.py` covers:

1. sequential feature snapshots apply one headroom hypothesis then KEEP it on measured improvement;
2. an open verification window prevents a second proposal/write;
3. measured regression triggers physical-style readback-verified rollback in the fake WING transport;
4. operator takeover enters HOLD without fighting the operator, and later snapshots remain blocked.

The focused GitHub Actions workflow includes this test file. No physical WING write is part of this software gate.

## Next live goal

After CI evidence is green, the next bounded gate is an explicitly declared hardware development session: one low-risk fader or PEQ hypothesis on the real WING, sequential feature verification, forced Critic rejection, verified physical rollback and second readback. Passing that gate is required before legacy AutoFader/AutoEQ write authority can be severed or archived.
