# ADR: read-surface-fixture-declaration-gate-20260605

## Context

Read-surface fixture evidence rows name test files and pytest functions that
support cached DTO coverage. The no-WING release report uses those rows to keep
console read awareness gaps visible. If a row references stale fixture metadata,
the report can overstate evidence quality even while remaining read-only.

## Options considered

1. Validate fixture file and test-name declarations in the no-WING schema.
2. Trust the builder-level unit test only.
3. Execute every declared test from the schema validator.

## Decision

Use option 1. The release schema will parse declared test files and verify that
each non-missing evidence row names existing `tests/...` files and pytest
functions.

## Why this won

The release schema is the proof boundary. AST-based declaration checks are
read-only, deterministic, and cheap, while still failing closed on stale
evidence metadata.

## Rejected alternatives

Builder-only validation is weaker because release summaries should be
self-verifying. Executing every declared test in the schema validator duplicates
the release command runner and makes the schema check heavier than needed.

## Implementation plan

1. Add a small AST helper that lists pytest function names in a file.
2. Validate fixture path shape and existence for each read-surface row.
3. Validate declared test names against the parsed files.
4. Reject stale declarations on `missing_fixture_evidence` rows.
5. Add corrupted-declaration tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py -q`
- `python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: future fixture renames will fail the release gate until evidence rows are
updated. That is acceptable because stale evidence should not support autonomy
promotion.

Rollback is removing the declaration parser/checks and corrupted-declaration
test.
