# Codex Proposal: Routing Sends Cache Freshness - 2026-06-05

## Short thesis

Add freshness evidence to routing/sends DTOs so downstream automation can reject stale routing state before proposing or applying matrix/send decisions.

## What I understood

The AI Mixing Engineer goal requires real-time routing awareness, but cached read DTOs are only useful when consumers can verify that the cache is current enough. Routing and send rows currently expose coverage, but not whether the observed values are fresh, stale, or untimestamped.

## Proposed solution

- Attach `_cache_freshness_for_field_addresses` to channel bus sends, channel main sends, channel matrix sends, and output route rows.
- Add an aggregate `_routing_sends_cache_addresses` helper and top-level snapshot `cache_freshness`.
- Surface `cache_freshness_status` in snapshot summary.
- Update inventory, fixture evidence, and tests.

## Likely files to touch

- `backend/operator_product_state.py`
- `backend/read_surface_inventory.py`
- `backend/read_surface_fixture_evidence.py`
- `tests/test_operator_product_state.py`
- `tests/test_read_surface_inventory.py`
- `tests/test_read_surface_fixture_evidence.py`
- `Docs/adr/routing-sends-cache-freshness-20260605.md`

## Alternatives considered

- Query routing/sends live to refresh state. Rejected because Product Layer snapshots must remain cached/read-only.
- Only add top-level freshness. Rejected because proposal builders need row-level evidence for one send or route.
- Promote route/send writes once freshness exists. Rejected because freshness is necessary but not sufficient for safe routing control.

## Risks

- Consumers may treat fresh cached state as full physical proof. The snapshot still remains non-autonomous proof with explicit physical fixture and diff blockers.

## Test plan

- Add focused routing/sends tests for fresh, missing, and stale freshness evidence.
- Run operator product-state, read-surface inventory, fixture evidence, registry/capability tests.
- Run `py_compile`, `git diff --check`, and the full backend suite.

## Where the other agent may disagree

Kimi may ask for a shared route graph verifier before adding freshness fields. That verifier is a later layer; freshness is a prerequisite evidence primitive and can land safely now.
