# Codex Proposal: Bus Main DCA Cache Freshness - 2026-06-05

## Short thesis

Add aggregate cache freshness to the bus/main/DCA Product Layer snapshot so balance automation can reject stale group/output state before making decisions.

## What I understood

The AI Mixing Engineer goal requires reliable group and output visibility. The current bus/main/DCA DTO exposes cached values and read coverage, but the surface still lists stale-cache timestamps as a blocker. A snapshot can be structurally complete yet too old to trust in a live mix.

## Proposed solution

- Add `_bus_main_dca_cache_addresses` for bus identity/basic/EQ/dynamics/main-send/assignment fields, main basic/EQ fields, and DCA identity/basic fields.
- Add top-level `cache_freshness` to `build_bus_main_dca_snapshot`.
- Surface `cache_freshness_status` in the snapshot summary.
- Update read-surface inventory, fixture evidence, and focused tests.

## Likely files to touch

- `backend/operator_product_state.py`
- `backend/read_surface_inventory.py`
- `backend/read_surface_fixture_evidence.py`
- `tests/test_operator_product_state.py`
- `tests/test_read_surface_inventory.py`
- `tests/test_read_surface_fixture_evidence.py`
- `Docs/adr/bus-main-dca-cache-freshness-20260605.md`

## Alternatives considered

- Query bus/main/DCA state live to refresh freshness. Rejected because Product Layer snapshots must remain cached/read-only.
- Add row-level freshness to every section now. Useful later, but aggregate freshness closes the named blocker with less surface churn.
- Promote bus/main/DCA apply after adding freshness. Rejected because freshness is necessary evidence, not write safety proof.

## Risks

- Consumers may treat fresh cached state as physical proof. The snapshot remains non-autonomous proof with physical fixture, membership, matrix-send, and parity blockers.

## Test plan

- Add focused bus/main/DCA snapshot tests for fresh, missing, and stale cache freshness.
- Run operator product-state, read-surface inventory, fixture evidence, registry/capability tests.
- Run `py_compile`, `git diff --check`, and the full backend suite.

## Where the other agent may disagree

Kimi may ask for full group membership fixtures before changing the inventory blocker. That remains a blocker; this patch only removes the stale-cache part once the DTO proves freshness.
