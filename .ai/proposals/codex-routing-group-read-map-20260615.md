# Codex Proposal: Routing / Group Read-only Topology Map

## Short Thesis

Add a transport-free topology map that normalizes cached/static routing,
bus/group/DCA, and matrix read rows before downstream route or group proposals
can claim full context.

## What I Understood

The Automixer goal requires full console-state awareness and future control of
routing, DCA, groups, and matrices. These surfaces can repatch or rebalance
broad mixer state, so current work must remain read-only until physical
readback and rollback proof exist.

## Proposed Solution

- Add `backend/routing_group_read_map.py`.
- Accept already captured Product Layer rows for routing/sends,
  bus/group/DCA, and matrix state.
- Normalize those rows into a topology edge list.
- Require fresh surface snapshots before a ready map claim.
- Block missing routing destructive-change markers and missing group/DCA
  membership evidence.
- Register the method as `shadow_only` and wire it into console read, routing,
  and DCA/group/matrix capability rows.

## Likely Files to Touch

- `backend/routing_group_read_map.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_routing_group_read_map.py`
- `tests/test_project_goal_capability_matrix.py`
- `tests/test_processing_registry.py`
- `.ai/project-layer/correction_methods_backlog.md`

## Alternatives Considered

- Expand route-diff proposals directly: rejected because the missing piece is
  read-only topology evidence, not another route-change recommendation.
- Query WING to build the map: rejected because this Goal-mode pass must stay
  no-WING and transport-free.

## Risks

- The default registry snapshot is intentionally incomplete because no live or
  cached rows are passed at registry build time.
- Consumers may need adapter work to feed real Product Layer rows into the map.

## Test Plan

- Compile touched Python files.
- Run focused routing/group map, read-surface, routing proposal, bus/group,
  registry, and capability matrix tests.
- Run full project pytest and no-WING release checks before calling the
  increment validated.

## Where The Other Agent May Disagree

Kimi may prefer a physical readback artifact first. The topology map is still
useful now because it defines the non-authoritative read-only evidence shape
that those future physical artifacts must satisfy.
