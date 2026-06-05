# ADR: read-surface-summary-drift-gate-20260605

## Context

`read_surface_fixture_evidence` exposes fixture coverage for console read
surfaces without querying hardware or executing fixtures. The no-WING release
check displays summary counts from that snapshot. For autonomy promotion, those
summary counts must be verified against row data so stale or malformed metadata
cannot hide remaining physical readback gaps.

## Options considered

1. Recompute summary facts in the no-WING schema validator.
2. Trust the snapshot builder summary.
3. Block releases until all physical readback fixtures exist.

## Decision

Use option 1. The release check will recompute row counts, status counts,
fixture-present counts, and physical-missing IDs from `surfaces`.

## Why this won

The release check is the proof boundary. Recomputing facts there keeps the gate
fail-closed even if a snapshot summary drifts or malformed metadata appears.

## Rejected alternatives

Trusting the summary is weaker evidence. Blocking all no-WING releases until
physical fixtures exist would stop incremental proof hardening and is larger
than this task.

## Implementation plan

1. Import the canonical read-surface fixture status set.
2. Validate that `surfaces` is present and row-like.
3. Recompute summary counts and physical-missing IDs from rows.
4. Reject unknown row statuses and non-integer summary counts.
5. Add a corrupted-summary unit test.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py -q`
- `python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk: stricter validation may fail if future snapshots change shape without a
schema migration. The violation names make that failure explicit.

Rollback is removing the recomputation checks and the corrupted-summary test.
