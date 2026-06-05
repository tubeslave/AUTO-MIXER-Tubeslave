# ADR: AutoPanner Apply Fail-Closed Boundary

## Context

`backend/auto_panner.py` computes pan decisions using instrument role, spectral centroid, dominant band, and symmetric distribution rules. This analysis is useful for the target AI mixing engineer. The legacy `apply_to_mixer()` method directly called mixer-client `set_channel_pan`, and `backend/auto_fader_v2/controller.py` had a pan-separation branch that also directly called `set_channel_pan`.

Pan writes are part of the quarantined pan/phase/delay write surface. They can damage mono compatibility, conflict with manual stage placement, or move sources without current-pan readback. Current live execution remains limited to supervised fader/gain.

## Options considered

1. Keep AutoPanner analysis but make direct pan apply fail closed.
2. Delete `apply_to_mixer()` immediately.
3. Keep controller pan separation live while blocking AutoPanner.
4. Promote pan writes into supervised auto mode now.

## Decision

Choose option 1. AutoPanner remains an analysis/proposal layer. `apply_to_mixer()` and controller pan separation return blocked apply evidence and perform no mixer-client mutation.

## Why this won

This preserves useful panning decisions while removing hidden direct pan writes. It aligns legacy panner behavior with `pan_spatial_proposal.py`, the write-surface inventory, and the current live executor boundary.

## Rejected alternatives

- Deleting the method could break callers without providing safe blocked evidence.
- Leaving controller pan separation live would preserve a bypass around the inventory.
- Promoting pan writes now would skip current-pan readback, mono compatibility, rollback, and physical proof.

## Implementation plan

- Add fail-closed policy fields to AutoPanner.
- Convert `apply_to_mixer()` into a no-mutation method returning `blocked_calls`.
- Convert controller pan separation into blocked evidence instead of direct `set_channel_pan` calls.
- Add `set_channel_pan` to static write-token detection and inventory.
- Update registry, capability matrix, processing inventory, tests, and council artifacts.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/auto_panner.py backend/auto_fader_v2/controller.py backend/processing_registry.py backend/write_surface_inventory.py backend/project_goal_capability_matrix.py tests/test_auto_panner.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_write_surface_inventory_static_scan.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_auto_panner.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_write_surface_inventory_static_scan.py -q`
- Runtime fake-mixer smoke test for no mutation.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: legacy callers that expected an integer apply count now receive a structured blocked result. This is intentional because direct pan writes are not approved. Rollback must not restore direct `set_channel_pan` calls; a live path requires a supervised proposal/apply contract with current-pan readback, mono compatibility proof, rollback values, and a separate ADR.
