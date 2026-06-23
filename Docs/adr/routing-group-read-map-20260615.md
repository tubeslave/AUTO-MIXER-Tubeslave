# ADR: Routing / Group Read-only Topology Map

Date: 2026-06-15
Status: accepted

## Context

The Automixer goal requires reading and eventually controlling routing, DCA,
groups, buses, and matrices. Existing route-diff and bus/group proposal layers
remain proposal-only, and physical readback contracts exist, but the correction
backlog still called out a missing read-only routing/group map.

## Options Considered

1. Leave Routing / Groups as `read_only_map_needed`.
2. Add a transport-free topology map over cached/static Product Layer rows.
3. Query WING or promote route/group writes.

## Decision

Use option 2. `routing_group_read_map` is a `shadow_only` method that normalizes
read rows into route, bus/group/DCA, matrix, and edge topology evidence. Missing
or stale surfaces, missing destructive-change markers, and missing membership
evidence keep the map incomplete.

## Why This Won

It advances full console-state awareness without expanding live authority. The
map gives routing and bus/group proposal layers a concrete evidence shape while
remaining read-only, non-queueable, and transport-free.

## Rejected Alternatives

- Adding another route proposal would not close the read-only map gap.
- Querying hardware or promoting writes is premature before WING read-only
  proof, physical fixtures, rollback, and manual review are complete.

## Implementation Plan

- Add `backend/routing_group_read_map.py`.
- Register it as a `shadow_only` processing method.
- Add it to console read, routing control, and DCA/group/matrix capability rows.
- Update correction backlog from `read_only_map_needed` to `read_only_map`.
- Add focused tests for read-only safety, ready/incomplete map states, registry
  snapshot visibility, and capability visibility.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- `PYTHONPATH=backend python -m pytest -q tests/test_routing_group_read_map.py tests/test_read_surface_inventory.py tests/test_routing_management_proposal.py tests/test_bus_group_balance_proposal.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`

## Risks and Rollback

The registry-level map is incomplete until real Product Layer rows are passed.
Rollback is removing the map module, registry entry, capability references,
tests, and documentation. No live transport code is touched.
