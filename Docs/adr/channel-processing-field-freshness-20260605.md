# ADR: channel-processing-field-freshness-20260605

## Context

Autonomous EQ, compression, and gate decisions depend on many cached console
fields. A section-level coverage flag can say that compressor data exists, but
it does not show whether the threshold, ratio, gain reduction, or makeup value
was recently observed. Without field freshness, stale cached processing state
can look stronger than it really is.

## Options considered

1. Keep only top-level channel-processing cache freshness.
2. Add section and field freshness metadata to cached processing DTOs.
3. Start with physical query fixtures and leave cached DTOs unchanged.

## Decision

Use option 2. Keep the snapshot read-only and cache-backed, but expose
field-level freshness for HPF, EQ, compressor, gate, and EQ-band fields.

## Why this won

It improves observability without contacting or mutating a console. Later
proposal builders can reject only the stale processing section instead of
treating the entire channel snapshot as equally trustworthy.

## Rejected alternatives

Option 1 is too coarse for autonomous processing. Option 3 is still required for
promotion, but it can build on a clearer DTO contract.

## Implementation plan

- Add a helper that maps fields to cached console addresses and reports per-field
  freshness.
- Attach freshness to channel-processing HPF, EQ, compressor, gate, and EQ-band
  read coverage.
- Update read-surface inventory wording to reflect the new cached freshness
  evidence while keeping physical proof blockers.
- Add regression tests for fresh and stale processing field timestamps.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/operator_product_state.py backend/read_surface_inventory.py tests/test_operator_product_state.py`
- Focused product-state/read-surface pytest.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: consumers may need to ignore the new nested freshness field until they use
it. Rollback is removing the helper, freshness keys, test assertions, inventory
wording, and this ADR. No live write behavior is affected.
