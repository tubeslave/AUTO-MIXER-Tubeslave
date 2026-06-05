# ADR: Adaptive Panner Send Fail-Closed Boundary

## Context

`backend/auto_panner_adaptive.py` computes multi-band adaptive panning decisions using feature extraction, instrument classification, genre templates, and smoothed pan gains. The analysis is useful for the target AI mixing engineer. The legacy `_send_osc()` method directly called mixer-client `set_channel_pan` when a mixer client was present.

Direct pan writes are part of the quarantined pan/phase/delay write surface. They can damage mono compatibility, conflict with manual stage placement, or move sources without current-pan readback. Current live execution remains limited to supervised fader/gain.

## Options considered

1. Keep adaptive panner analysis but make `_send_osc()` fail closed.
2. Delete `_send_osc()` immediately.
3. Leave adaptive panner lab-only without changing runtime behavior.
4. Promote adaptive pan writes into supervised auto mode now.

## Decision

Choose option 1. Adaptive panner remains an analysis/proposal layer. `_send_osc()` stays for compatibility but returns blocked apply evidence and performs no mixer-client mutation.

## Why this won

This preserves useful adaptive pan decisions while removing a hidden direct write path. It aligns adaptive panner behavior with `AutoPanner.apply_to_mixer()`, `pan_spatial_proposal.py`, the write-surface inventory, and the current live executor boundary.

## Rejected alternatives

- Deleting the method could break callers without providing safe blocked evidence.
- Leaving the lab path untouched would preserve a write bypass whenever a mixer client is supplied.
- Promoting pan writes now would skip current-pan readback, mono compatibility, rollback, and physical proof.

## Implementation plan

- Add fail-closed policy and blocked-write telemetry to AutoPannerController.
- Convert `_send_osc()` into a no-mutation method returning `blocked_calls`.
- Expose blocked write telemetry through `get_status()`.
- Update registry, capability matrix, processing inventory, and tests.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/auto_panner_adaptive.py backend/processing_registry.py backend/write_surface_inventory.py backend/project_goal_capability_matrix.py tests/test_auto_panner_adaptive.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_auto_panner_adaptive.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_write_surface_inventory_static_scan.py -q`
- Runtime fake-mixer smoke test for no mutation.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: legacy callers that expected `_send_osc()` to write pan values now receive a structured blocked result. This is intentional because direct pan writes are not approved. Rollback must not restore direct `set_channel_pan` calls; a live path requires a supervised proposal/apply contract with current-pan readback, mono compatibility proof, rollback values, and a separate ADR.
