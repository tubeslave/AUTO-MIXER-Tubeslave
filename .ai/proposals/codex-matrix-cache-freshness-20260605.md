# Codex Proposal: Matrix Cache Freshness - 2026-06-05

## Short thesis

Add aggregate cache freshness to the matrix Product Layer snapshot so matrix-aware automation can reject stale output state before making decisions.

## What I understood

The AI Mixing Engineer goal requires reliable matrix visibility. The current matrix DTO exposes cached values and section coverage, but the surface-level inventory still lists stale-cache timestamps as a blocker. A snapshot can be complete yet too old to trust during a show.

## Proposed solution

- Add `_matrix_cache_addresses` for matrix basic controls, EQ, dynamics, pre insert, and post insert addresses.
- Add top-level `cache_freshness` to `build_matrix_snapshot`.
- Surface `cache_freshness_status` in matrix snapshot summary.
- Update inventory, fixture evidence, and focused tests.

## Likely files to touch

- `backend/operator_product_state.py`
- `backend/read_surface_inventory.py`
- `backend/read_surface_fixture_evidence.py`
- `tests/test_operator_product_state.py`
- `tests/test_read_surface_inventory.py`
- `tests/test_read_surface_fixture_evidence.py`
- `Docs/adr/matrix-cache-freshness-20260605.md`

## Alternatives considered

- Query matrix state live to refresh freshness. Rejected because Product Layer snapshots must remain cached/read-only.
- Add per-field freshness only to dynamics. Already done, but not enough for whole matrix state.
- Promote matrix writes once freshness exists. Rejected because freshness is necessary evidence, not write proof.

## Risks

- Consumers may treat fresh cached state as physical proof. The snapshot still remains non-autonomous proof with explicit physical fixture, parity, and route diff blockers.

## Test plan

- Add focused matrix snapshot tests for fresh, missing, and stale cache freshness.
- Run operator product-state, read-surface inventory, fixture evidence, registry/capability tests.
- Run `py_compile`, `git diff --check`, and the full backend suite.

## Where the other agent may disagree

Kimi may prefer row-level freshness for every EQ band and insert section before changing the surface blocker. This patch adds aggregate freshness first because it is the fail-closed evidence downstream proposal builders need before trusting the matrix snapshot at all.
