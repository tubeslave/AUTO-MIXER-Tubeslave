# ADR: AutoFX Apply Fail-Closed Boundary

## Context

`backend/auto_fx.py` builds useful live-style FX plans: shared stereo returns, tempo delay, mod doubler, and post-fader sends. It also had a legacy `apply_to_mixer()` method that called mixer-client FX/send methods directly. Direct FX, insert, and send writes are quarantined, and the current live executor remains limited to supervised fader/gain.

FX writes are broad mixer-state changes. A wrong FX model, wet/dry level, send level, or return level can reduce intelligibility, create feedback risk, or hide manual engineer choices. These writes need send/return readback, feedback checks, rollback proof, and operator-reviewed promotion before live apply.

## Options considered

1. Keep AutoFX planning but make `apply_to_mixer()` fail closed.
2. Delete `apply_to_mixer()`.
3. Keep direct apply for lab/offline mode.
4. Promote AutoFX directly into assisted live control.

## Decision

Choose option 1. `AutoFXPlanner` remains a proposal-only planning layer. `apply_to_mixer()` stays as a compatibility API but returns blocked apply evidence and performs no mixer-client mutation.

## Why this won

This keeps the useful FX planning logic while removing an unsafe direct-write path. It aligns `auto_fx.py` with `fx_send_proposal.py` and the write-surface inventory: FX behavior can be inspected and reviewed, but direct apply stays quarantined.

## Rejected alternatives

- Deleting `apply_to_mixer()` could break unknown legacy callers without giving them a safe no-op result.
- Lab/offline direct apply still risks accidental live transport use when a mixer client is passed.
- Direct assisted promotion is premature without physical readback and rollback evidence.

## Implementation plan

- Add fail-closed policy fields to `AutoFXPlanner`.
- Convert `apply_to_mixer()` into a no-mutation method returning `blocked_calls`.
- Build intended FX/send command records without invoking mixer-client methods.
- Update registry, write-surface inventory, capability matrix, processing inventory, and tests.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/auto_fx.py backend/processing_registry.py backend/write_surface_inventory.py backend/project_goal_capability_matrix.py tests/test_auto_fx.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_auto_fx.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_write_surface_inventory_static_scan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- Runtime fake-mixer smoke test for no mutation.

## Risks and rollback

Risk: legacy callers that expected FX to be applied now only receive blocked evidence. This is intentional. Rollback must not restore direct FX writes; if live FX apply is needed, add a supervised proposal/apply contract with physical readback, feedback checks, bounded return/send limits, rollback values, and a separate ADR.
