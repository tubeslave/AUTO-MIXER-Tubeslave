# ADR: user-presets-read-dto-20260604

## Context

User presets are required for the long-term autonomous console-control target,
but recalling or storing presets can rewrite broad console state. The inventory
previously marked user presets as missing.

## Options considered

1. Keep user presets as not available.
2. Add a cached-only Product Layer user-preset DTO.
3. Add preset recall/store proposal or apply support.

## Decision

Use option 2. Add `build_user_presets_snapshot` as cached-only read coverage and
keep recall/store blocked.

## Why this won

The DTO gives the cockpit a stable read contract for known preset slots without
querying hardware, recalling presets, storing presets, or widening live writes.

## Rejected alternatives

Option 1 leaves an avoidable gap in read coverage. Option 3 is unsafe because
preset contents, scoped diff, stale-cache timestamps, and rollback proof are
incomplete.

## Implementation plan

- Add cached user preset catalog helpers.
- Add `get_user_presets_snapshot`.
- Update read surface inventory.
- Add focused cached/missing tests and handler round-trip coverage.

## Test plan

- `PYTHONPATH=backend python -m pytest tests/test_operator_product_state.py tests/test_read_surface_inventory.py -q`
- Broader product-state/readiness tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `git diff --check`

## Risks and rollback

Risk: UI users may overread cached preset names as physical proof. Mitigation:
publish read coverage and blocked recall/store flags. Rollback is removing the
builder, handler, tests, and inventory updates.
