# TUB-265 Replay Failure Recovery Validation

## Scope

Heartbeat validation for [TUB-265](http://localhost:3000/032f8339-d6e9-4f49-b10f-c257814dfa31/issues/TUB-265): confirm whether the repository already contains a replay-safe intelligence failure recovery simulation suite for the Analyzer -> Critic -> Decision -> Executor graph, and verify that the suite remains dry-run-only with no accidental live transport.

No production code was modified in this heartbeat.

## Verdict

The core replay-safe intelligence recovery suite is present and materially working.

Verified coverage exists for:

- deterministic proposal/ranking checkpoints;
- replay checkpoint manifests and rewind restoration;
- executor simulation and rollback restoration;
- policy/coherence validation with failure injection;
- replay benchmark baselines and drift detection;
- replay critic harness artifact generation;
- replay-safe transport blocking and Tier A live-apply gating.

The targeted verification run passed the core graph suite. The only failures were three direct-write inventory tests in arrangement automation / auto-fader paths, which are outside the core replay intelligence graph and should not block closing the replay recovery suite itself.

## Verified Test Run

Command:

```bash
python3 -m pytest -q \
  tests/test_replay_policy_validator.py \
  tests/test_replay_executor.py \
  tests/test_replay_restore_loader.py \
  tests/test_replay_intelligence_benchmark_suite.py \
  tests/test_replay_harness.py \
  tests/test_live_apply_contract.py \
  tests/test_direct_write_inventory.py
```

Observed result:

- 78 passed
- 3 failed

## What The Current Suite Proves

### 1. Replay failure injection exists

- `tests/test_replay_policy_validator.py`
  - tampers replay signatures;
  - flips `dry_run_only` and `live_mixer_writes`;
  - injects executor/live-write mismatches;
  - detects selected-proposal drift against a baseline.
- `tests/test_replay_intelligence_benchmark_suite.py`
  - compares deterministic baselines against a drifted candidate graph.

### 2. Deterministic recovery validation exists

- `tests/test_replay_restore_loader.py`
  - restores typed rewind context from a dry-run manifest;
  - rejects live-write manifests.
- `tests/test_replay_executor.py`
  - validates deterministic trace signatures;
  - validates rollback to an intermediate state;
  - validates initialization from rewind context.

### 3. Replay corruption simulation exists

- `tests/test_replay_policy_validator.py`
  - analyzer signature corruption;
  - timeline policy corruption;
  - executor dry-run corruption;
  - selected-proposal mismatch.
- `tests/test_replay_report_traceability.py`
  - verifies correlation IDs survive into report artifacts.

### 4. Intelligence graph resilience criteria exist

- `tests/test_replay_intelligence_benchmark_suite.py`
  - stable manifest metrics;
  - stable ranking signature;
  - stable executor trace signature;
  - rewind correlation inventory;
  - drift detection surface.

### 5. Replay-only governance is enforced

- `ai_mixing_pipeline/decision_layer/replay_restore_loader.py`
  - rejects non-dry-run manifests.
- `ai_mixing_pipeline/decision_layer/replay_policy_validator.py`
  - fails on live-write metadata drift and executor dry-run violations.
- `tests/test_replay_harness.py`
  - installs socket/python-osc transport blockers;
  - proves dry-run queue reconstruction;
  - proves unsupported live-apply operations are blocked without mutation.
- `tests/test_live_apply_contract.py`
  - proves manual live-apply defaults to dry-run;
  - proves confirmation does not bypass dry-run;
  - proves panic stop blocks real send;
  - proves unsupported operations fail closed.

## Failure Analysis

The three failing tests are:

1. `tests/test_direct_write_inventory.py::test_arrangement_automation_inventory_live_enabled_enforces_fader_ceiling`
2. `tests/test_direct_write_inventory.py::test_arrangement_automation_replay_safe_offset_plan_applies_through_live_apply_gate`
3. `tests/test_direct_write_inventory.py::test_auto_fader_inventory_apply_auto_balance_enforces_min_max_fader_clip`

These failures indicate direct-write inventory drift, not replay-graph recovery failure.

### Arrangement automation drift

- `backend/arrangement_automation/controller.py:294-338`
- Live apply only occurs when:
  - `analysis_only_mode` is false;
  - `live_apply_enabled` is true;
  - `confirm_live_apply` is true;
  - mixer is connected.
- The failing ceiling test config omits `confirm_live_apply`, so the defaulting path is part of the behavior under test.
- The replay reapply mismatch shows the recorded plan and the direct `_apply_safe_offsets()` replay path are not reconstructing the same applied value in the current fixture state.

### Auto-fader drift

- `backend/auto_fader.py:2011-2061`
- `apply_auto_balance()` prefers mixer readback from `get_channel_fader()` over `state.current_fader`.
- In the failing test fixture, the fake mixer default read is `-5.0 dB` while `state.current_fader` was set to `-55.0 dB`.
- The observed `-25.0 dB` result therefore comes from `-5.0 + (-20.0)` and then clipping, not from the expected `-55.0 + (-20.0) -> -60.0`.

## Safety Claim Boundary

This heartbeat supports the following claim:

- the replay intelligence failure-recovery suite is dry-run-only and operational for the core Analyzer -> Critic -> Decision -> Executor graph.

This heartbeat does not support the following stronger claim:

- every non-graph automation path has perfectly replay-stable direct-write inventory behavior.

Those are separate QA surfaces.

## Recommended Follow-Up

1. Split arrangement automation confirmation-default coverage from fader-ceiling coverage so the expected safety gate is explicit.
2. Add a dedicated replay fixture for arrangement `_apply_safe_offsets()` that persists the base-fader context needed for deterministic reapply.
3. Decide whether `AutoFaderController.apply_auto_balance()` should trust mixer readback first or fixture state first, then align the test and contract.

## Disposition Recommendation

`TUB-265` can be closed as complete for the replay-safe intelligence failure recovery suite itself.

Follow-up work, if desired, should be tracked separately under direct-write inventory stabilization for arrangement automation and auto-fader paths rather than under the core replay intelligence recovery task.
