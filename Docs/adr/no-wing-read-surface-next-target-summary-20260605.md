# ADR: no-wing-read-surface-next-target-summary-20260605

## Context

Read-surface proof quality now has a visible next target in Product Layer and
Settings. The no-WING release artifact should also expose that target so the
proof source, Product Layer, and cockpit agree.

## Options considered

1. Add `next_physical_readback_surface_id` to read-surface fixture evidence
   snapshots and no-WING summaries.
2. Keep deriving the target downstream.
3. Build a full remediation priority planner.

## Decision

Use option 1. The summary records the first physical readback gap as the next
target, and the release schema fails if the target does not match the gap list.

## Why this won

It makes the proof target auditable at the artifact source without touching
hardware or widening live control authority.

## Rejected alternatives

Downstream derivation can drift between tools. A priority planner is larger than
this evidence slice and should wait for an agreed physical proof workflow.

## Implementation plan

1. Add the target field in `backend/read_surface_fixture_evidence.py`.
2. Validate target/list consistency in `tools/no_wing_release_check.py`.
3. Expose the target in JSON and Markdown summaries.
4. Add focused tests for snapshot, schema drift, and summary output.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_fixture_evidence.py tests/test_no_wing_release_check.py -q`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: a release summary target may be mistaken for approval to run hardware
readback. The field remains read-only evidence and does not change execution
policy.

Rollback is removing the field, validation, Markdown output, tests, and this
ADR.
