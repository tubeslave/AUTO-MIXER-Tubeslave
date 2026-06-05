# ADR: read-surface-contract-gap-consistency-gate-20260605

## Context

The read-surface fixture evidence snapshot distinguishes cached DTO fixture
coverage from missing physical readback proof. The no-WING release gate already
validates summary drift and fixture declarations, but it should also reject
contradictory row status, contract, and gap combinations.

## Options considered

1. Add contract/gap consistency checks to the no-WING schema validator.
2. Trust builder-level status generation.
3. Promote cached DTO rows to physical proof when tests exist.

## Decision

Use option 1. The release schema will validate row-level status invariants
without executing tests or touching hardware.

## Why this won

It strengthens the proof boundary while preserving the safety posture: cached
fixtures remain useful evidence, but they cannot imply physical console proof or
Full Autonomous readiness.

## Rejected alternatives

Builder-only validation is too weak for release proof. Promoting cached DTO rows
to physical proof is unsafe and contradicted by the current no-WING evidence.

## Implementation plan

1. Validate required row identity fields.
2. Validate `covered_contracts` and `remaining_gaps` shape.
3. Enforce status-specific contract/gap invariants.
4. Add corrupted row tests.
5. Keep summary output read-only and non-autonomous.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py -q`
- `python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: future evidence statuses need schema migration. Rollback is removing the
row consistency checks and their corrupted-row test.
