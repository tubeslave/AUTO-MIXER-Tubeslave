# Routing / Group Read-only Topology Map

Date: 2026-06-15
Status: implemented and validated

## Summary

Routing / Groups now has a concrete read-only map artifact:

- `routing_group_read_map` normalizes cached/static routing, bus/group/DCA, and
  matrix rows into topology edges.
- Missing/stale read surfaces keep the map `read_map_incomplete`.
- Missing routing destructive-change markers block a ready map claim.
- Missing group/DCA membership evidence blocks a ready map claim.
- The method is registered as `shadow_only`.
- Routing and DCA/group/matrix capability rows now require topology-map
  evidence before proposal promotion.

## Safety Boundary

This does not contact WING, OSC, MIDI, or mixer clients. It does not create
route, group, DCA, matrix, send, or queue writes, and it does not grant live
authority.

## Validation

- `PYTHONPATH=backend python -m py_compile backend/routing_group_read_map.py backend/processing_registry.py backend/project_goal_capability_matrix.py tests/test_routing_group_read_map.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py`
  - PASS
- `PYTHONPATH=backend python -m pytest -q tests/test_routing_group_read_map.py tests/test_read_surface_inventory.py tests/test_routing_management_proposal.py tests/test_bus_group_balance_proposal.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py`
  - `104 passed`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  - `3439 passed, 9 skipped`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  - `status: passed`
  - `live_authority_granted: false`
  - `full_autonomous_ready: false`
  - `readiness_closure_allowed_by_packet: false`
- `PYTHONPATH=backend python tools/memory/memory_lint.py`
  - PASS
- `/opt/homebrew/bin/git diff --check`
  - PASS
