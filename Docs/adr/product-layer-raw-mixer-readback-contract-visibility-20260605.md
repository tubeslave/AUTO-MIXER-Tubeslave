# ADR: product-layer-raw-mixer-readback-contract-visibility-20260605

## Context

The no-WING release summary validates the raw mixer-state physical readback
contract. Product Layer proof evidence reads the same summary for read-surface
fixture quality, but it did not expose contract quality to cockpit operators.

## Options considered

1. Add contract quality to the existing supervised proof evidence snapshot.
2. Leave visibility only in release artifacts.
3. Add a new Product Layer endpoint.

## Decision

Use option 1. Product Layer normalizes the raw mixer-state readback contract
from the no-WING summary and the frontend renders it as read-only Module Status
and Settings proof evidence.

## Why this won

It preserves the current evidence path while closing an operator-visibility gap.
The same artifact now explains fixture gaps, remediation action, and artifact
contract status without adding hardware access.

## Rejected alternatives

Release-only visibility hides contract drift from the cockpit. A new endpoint
would add surface area without a new source of truth.

## Implementation plan

1. Normalize raw mixer contract fields in Product Layer proof quality.
2. Add frontend adapter tags and a dedicated Module Status row.
3. Add Settings proof inspector rows for contract status, gap, authority, and
   target consistency.
4. Update backend and frontend/e2e contract tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q`
- `npm --prefix frontend run build`
- `npm --prefix frontend test -- --watch=false`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: contract visibility could be misread as proof completion. Mitigation:
Product Layer normalizes live authority and Full Autonomous readiness to false
and keeps the physical fixture gap visible.

Rollback is removing the Product Layer normalization, frontend display fields,
tests, and this ADR. No live transport path is added.
