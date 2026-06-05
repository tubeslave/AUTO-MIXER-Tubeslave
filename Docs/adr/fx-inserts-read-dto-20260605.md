# ADR: fx-inserts-read-dto-20260605

## Context

FX rack state and insert assignments are central to autonomous mix control, but
they can alter the processing chain. Existing handlers include query and
quarantined write paths, while the Product Layer lacked a cached read DTO.

## Options considered

1. Keep FX/insert state only in transport helpers.
2. Add a cached-only Product Layer DTO.
3. Add FX/insert proposal or live apply support.

## Decision

Use option 2. Add `build_fx_inserts_snapshot` and keep FX/insert apply blocked.

## Why this won

The cockpit gains stable read coverage for FX slots and insert assignments
without querying hardware or creating a new console write path.

## Rejected alternatives

Option 1 leaves the Product Layer blind to a goal-critical surface. Option 3 is
unsafe without parameter schemas, physical readback fixtures, scoped diff, and
rollback proof.

## Implementation plan

- Add cached FX slot rows with raw cached parameters.
- Add cached insert rows for channel, bus, main, and matrix targets.
- Register `get_fx_inserts_snapshot`.
- Update read surface inventory and focused tests.

## Test plan

- `PYTHONPATH=backend python -m pytest tests/test_operator_product_state.py tests/test_read_surface_inventory.py -q`
- Broader server/registry/readiness tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `git diff --check`

## Risks and rollback

Risk: raw FX parameter values can be misread as semantic controls. Mitigation:
coverage flags and blockers require schema mapping before promotion. Rollback
is removing the builder, handler, tests, and inventory updates.
