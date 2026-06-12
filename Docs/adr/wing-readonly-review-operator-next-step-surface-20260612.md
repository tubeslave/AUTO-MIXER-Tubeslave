# ADR: WING read-only review operator next step surface

## Context

The WING read-only runner writes `operator_next_step` when it blocks before a
backend WebSocket. The review layer previously preserved violations but not the
operator-facing repair instruction.

## Options Considered

1. Leave next-step guidance only in runner and no-WING artifacts.
2. Add one top-level review next step.
3. Preserve `operator_next_step` per artifact review row.

## Decision

Preserve `operator_next_step` per artifact review row.

## Why This Won

Artifact rows are the review unit. Keeping the field on each row avoids mixing
guidance from different runs and keeps invalid artifacts fail-closed while still
telling the operator what to replace or regenerate.

## Rejected Alternatives

Top-level guidance was rejected because multiple artifact paths can be reviewed
together. Leaving the field out was rejected because manual review reports would
hide the runner's specific repair instruction.

## Implementation Plan

- Extend `WingReadonlyPreflightArtifactRecord`.
- Populate the field from summary payload validation evidence.
- Include the field in review JSON and Markdown rendering.
- Add a blocked expected-plan regression test.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_evidence.py`
- Adjacent WING/no-WING tests.
- Full test suite before commit.

## Risks And Rollback

The field must remain diagnostic. Rollback is to remove the field from the
record and renderer; candidate acceptance gates are unchanged.
