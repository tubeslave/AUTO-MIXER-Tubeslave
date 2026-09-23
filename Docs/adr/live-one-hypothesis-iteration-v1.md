# Live One-Hypothesis Iteration v1

Date: 2026-09-23
Status: implemented in `backend/live_runtime`, software evidence passed; physical WING HIL still required

## Decision

The live runtime may have at most one applied musical/control hypothesis awaiting perceptual verification at a time.

Canonical sequence:

`Director hypothesis -> authoritative apply/readback -> short non-blocking verify window -> live Critic -> KEEP or verified rollback`

`backend/live_runtime/iteration.py` owns this orchestration. It does not own OSC addresses, WING transport, studio editing, mastering or legacy AutoFOH decision policy.

## Runtime contract

`IterationCoordinator.start()`:
- rejects a second hypothesis while one is awaiting verification;
- executes through the supplied canonical runtime (`LiveSoundcheckService`/`LiveControlPlane` contract);
- does not open a Critic window when the active mode/policy blocks the write;
- if transport wrote but immediate mixer readback mismatches, it performs verified rollback immediately rather than treating that state as a valid experiment;
- enters HOLD if that restoration cannot be verified.

`IterationCoordinator.verify()`:
- is non-blocking and returns `VERIFY_WAIT` until the configured verification window has elapsed;
- delegates acoustic/perceptual judgment to the new live Critic (`decision_engine.verify` by default);
- keeps an improved hypothesis;
- uses the canonical verified rollback path on regression;
- enters HOLD if rollback fails.

## Manual-touch priority

`operator_took_control` is not treated as permission for an autonomous rollback. Once the operator has intervened, rollback could undo the operator's corrective move. The coordinator therefore stops the iteration and enters HOLD without issuing another mixer write.

This implements the `manual-touch priority` rule in `live-soundcheck-pipeline-v2.md` and avoids an authority fight between human and automation.

## Legacy audit

`backend/autofoh_evaluation.py` was inspected before reuse.

Classification:
- its acknowledgement that raw input channels cannot prove post-console acoustic effect is retained as design knowledge;
- any reusable detector/metric evidence remains **ADAPT** only after validation against the realtime feature stream;
- its `PendingActionEvaluation`, legacy typed-action rollback construction and proxy test-policy orchestration are **ARCHIVE** candidates once imports are severed, because `live_runtime` now owns apply/readback/verify/rollback semantics;
- no legacy AutoFOH decision or rollback implementation is imported by the new coordinator.

## Test evidence

Focused CI covers:
- exactly one in-flight hypothesis;
- non-blocking verification-window gating;
- KEEP after intended metric improvement;
- verified rollback after Critic rejection;
- operator-touch HOLD without autonomous rollback;
- immediate rollback after apply/readback mismatch;
- HOLD after failed rollback;
- blocked/observe write never opening a verification window.

The `Stem Offline Test` workflow now runs on `live-soundcheck-renovation` and includes `tests/test_live_iteration_coordinator.py`.

## HIL gate

This software proof is not sufficient to archive legacy AutoFader/AutoEQ/AutoFOH evaluation code. Physical WING evidence is still required:

1. explicitly enter BENCH_TEST;
2. apply one bounded reversible fader or PEQ hypothesis;
3. capture fresh before/write/readback evidence;
4. feed a deliberately failing Critic observation;
5. verify physical rollback and second readback;
6. confirm operator touch causes HOLD and no compensating write.

Only after runtime/import references are severed and this HIL gate passes may legacy evaluation/write authority move from ARCHIVE candidate toward removal.
