# Codex Proposal: FX Inserts Cache Freshness - 2026-06-05

## Short thesis

Add aggregate cache freshness to the FX/inserts Product Layer snapshot so effect automation can reject stale rack or insert state before making decisions.

## What I understood

The AI Mixing Engineer goal includes automatic FX control and FX send setup. The current FX/inserts DTO exposes cached values and coverage, but the surface still lists stale-cache timestamps as a blocker. Freshness is required before any proposal builder can safely reason about existing FX state.

## Proposed solution

- Add `_fx_inserts_cache_addresses` for FX slot metadata, parameter cache addresses, and pre/post insert assignments across channel, bus, main, and matrix targets.
- Add top-level `cache_freshness` to `build_fx_inserts_snapshot`.
- Surface `cache_freshness_status` in snapshot summary.
- Update read-surface inventory, fixture evidence, and focused tests.

## Likely files to touch

- `backend/operator_product_state.py`
- `backend/read_surface_inventory.py`
- `backend/read_surface_fixture_evidence.py`
- `tests/test_operator_product_state.py`
- `tests/test_read_surface_inventory.py`
- `tests/test_read_surface_fixture_evidence.py`
- `Docs/adr/fx-inserts-cache-freshness-20260605.md`

## Alternatives considered

- Query FX and insert state live to refresh freshness. Rejected because Product Layer snapshots must remain cached/read-only.
- Add row-level freshness for every FX parameter and insert row now. Useful later, but aggregate freshness closes the named blocker with less churn.
- Promote FX or insert apply after adding freshness. Rejected because freshness is necessary evidence, not write safety proof.

## Risks

- Consumers may treat fresh cached state as physical proof. The snapshot remains non-autonomous proof with physical fixture, semantic schema, scoped diff, and safety blockers.

## Test plan

- Add focused FX/inserts snapshot tests for fresh, missing, and stale cache freshness.
- Run operator product-state, read-surface inventory, fixture evidence, registry/capability tests.
- Run `py_compile`, `git diff --check`, and the full backend suite.

## Where the other agent may disagree

Kimi may ask for FX model parameter schemas before removing any blocker. That remains a blocker; this patch only removes the stale-cache part once the DTO proves freshness.
