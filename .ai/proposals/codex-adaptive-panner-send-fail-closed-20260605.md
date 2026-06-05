# Codex Proposal: Adaptive Panner Send Fail-Closed Boundary

## Short thesis

Keep the adaptive panner's multi-band pan analysis, but make its legacy `_send_osc()` path fail closed so adaptive pan commands become blocked evidence instead of mixer mutations.

## What I understood

The project goal includes automatic panning and console pan control. The repository already has proposal-only pan layers, and recent work closed `AutoPanner.apply_to_mixer()` plus controller pan separation. `backend/auto_panner_adaptive.py` still had a separate `_send_osc()` branch that could call `set_channel_pan` whenever a mixer client was supplied.

## Proposed solution

- Add fail-closed policy and blocked-write telemetry to `AutoPannerController`.
- Reset write telemetry at the start of `process()`.
- Convert `_send_osc()` into a no-mutation method returning:
  - `applied=False`;
  - `calls=[]`;
  - `blocked_calls` with would-be `set_channel_pan` commands;
  - `mutates_mixer=False`.
- Expose latest blocked write telemetry through `get_status()`.
- Update registry, write-surface inventory, capability matrix, processing inventory, and tests.

## Likely files to touch

- `backend/auto_panner_adaptive.py`
- `tests/test_auto_panner_adaptive.py`
- `backend/processing_registry.py`
- `backend/write_surface_inventory.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_processing_registry.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/PROCESSING_METHODS_INVENTORY.md`
- `Docs/adr/adaptive-panner-send-fail-closed-20260605.md`

## Alternatives considered

- Delete `_send_osc()`. Rejected because a fail-closed compatibility method is safer for unknown callers.
- Leave adaptive panner as lab-only. Rejected because the direct write path would still exist when a mixer client is passed.
- Promote adaptive pan writes now. Rejected because current-pan readback, mono checks, rollback proof, and physical validation are not complete.

## Risks

- Legacy callers may have expected `_send_osc()` to write pan values.
- The method name remains historical and can mislead readers unless tests and inventory stay explicit.
- Blocked evidence still contains write command names, so static scan and inventory coverage must remain in place.

## Test plan

- Compile touched Python modules.
- Run adaptive panner, registry, capability matrix, and write-surface tests.
- Run a runtime fake-mixer smoke test proving no `set_channel_pan` calls occur.
- Run the full backend suite.

## Where the other agent may disagree

Kimi may prefer removing `_send_osc()` entirely. I would keep it fail-closed for compatibility now and consider a later rename after all callers are inventoried.
