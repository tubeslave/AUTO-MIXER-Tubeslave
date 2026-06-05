# Codex Proposal: Matrix Dynamics Read DTO - 2026-06-05

## Short thesis

Add cached matrix dynamics visibility to the Product Layer so master/matrix decisions can inspect dynamics state without gaining live write authority.

## What I understood

The target AI Mixing Engineer must analyze matrices and output processing before it can safely optimize master/output paths. The current matrix DTO covers basic controls, EQ, and inserts, but not dynamics. That leaves a blind spot in the read model even though dynamics can strongly affect output loudness and headroom.

## Proposed solution

- Add a cached matrix dynamics section to `_matrix_surface_row`.
- Read only cached `/mtx/N/dyn/*` addresses.
- Include coverage flags and a DTO payload similar to existing bus dynamics.
- Update inventory, fixture evidence, and focused tests.

## Likely files to touch

- `backend/operator_product_state.py`
- `backend/read_surface_inventory.py`
- `backend/read_surface_fixture_evidence.py`
- `tests/test_operator_product_state.py`
- `tests/test_read_surface_inventory.py`
- `tests/test_read_surface_fixture_evidence.py`
- `Docs/adr/matrix-dynamics-read-dto-20260605.md`

## Alternatives considered

- Query matrix dynamics from WING. Rejected because Product Layer snapshots must remain cached/read-only.
- Leave dynamics as a blocker. Rejected because cached dynamics state can be normalized now without write risk.
- Promote matrix dynamics apply. Rejected because supervised proof only covers Product Layer `fader` and `gain`.

## Risks

- Consumers may treat cached matrix dynamics as physical proof. The snapshot remains partial autonomous evidence with explicit coverage and safety flags.

## Test plan

- Add focused matrix snapshot tests for cached dynamics and missing dynamics.
- Run operator product-state, read-surface inventory, fixture evidence, registry/matrix tests.
- Run `py_compile`, `git diff --check`, and the full backend suite.

## Where the other agent may disagree

Kimi may ask to build a generic dynamics section helper shared by bus and matrix. That is reasonable later, but this small patch avoids broad refactors in a dirty worktree.
