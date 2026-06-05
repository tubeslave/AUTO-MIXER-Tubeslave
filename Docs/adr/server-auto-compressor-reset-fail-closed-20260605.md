# ADR: Server Auto Compressor Reset Fail Closed - 2026-06-05

## Context

`AutoMixerServer._reset_compressor_on_channel` reset compressor state before starting Auto Compressor. Public WING Auto Compressor entrypoints are already blocked by the deployment boundary, and controller-level compressor apply paths already emit blocked evidence. The helper still contained direct `set_compressor_on` and `set_compressor` calls.

The only proven live-write path remains supervised Product Layer `fader` and `gain`. Compressor reset/apply requires fresh dynamics readback, makeup/headroom proof, rollback values and hardware validation before live promotion.

## Options considered

1. Convert the server reset helper into blocked evidence.
2. Leave the helper unchanged because public WING routes are quarantined.
3. Delete the helper outright.
4. Promote compressor reset to live supervised apply now.

## Decision

Choose option 1. `_reset_compressor_on_channel` now returns structured blocked evidence and does not mutate mixer clients.

## Why this won

It removes a hidden direct write while preserving the reset intent as evidence for future operator proposals. The change is small and consistent with the existing compressor fail-closed posture.

## Rejected alternatives

- Leaving the helper unchanged was rejected because private/internal calls could bypass the deployment boundary.
- Deleting the helper was rejected because the reset parameters are useful for review and replay.
- Live promotion was rejected because compressor proof requirements are not met.

## Implementation plan

1. Add a blocked reset evidence builder in `backend/server.py`.
2. Change `_reset_compressor_on_channel` to return blocked evidence.
3. Add direct server test coverage for the helper.
4. Update inventory, capability matrix, processing inventory and registry tests.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/server.py backend/write_surface_inventory.py backend/project_goal_capability_matrix.py tests/test_server.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_server.py::TestWingDeploymentWriteBoundary::test_auto_compressor_reset_helper_is_fail_closed_evidence tests/test_processing_registry.py::test_write_surface_inventory_covers_known_legacy_write_paths tests/test_project_goal_capability_matrix.py::test_compression_is_tracked_as_transport_free_proposal_layer -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: a legacy caller expecting reset mutation now receives blocked evidence. This is intentional under the current no-live-compressor-write policy.

Rollback must not restore direct compressor writes. A future live reset needs the same supervised proposal/apply proof as compressor apply: readback, rollback, bounded makeup/headroom checks and hardware validation.
