# ADR: source-identity-confidence-proposal-20260605

## Context

The project goal requires automatic source type classification. Existing name
and ML classifiers can produce confidence scores, but downstream processing
families need a shared, operator-visible way to handle unknown, low-confidence,
or ambiguous identities before trusting source-specific assumptions.

## Options considered

1. Leave classifier confidence as raw metadata only.
2. Add a proposal-only source identity review layer.
3. Let downstream processing modules each implement their own confidence gates.
4. Auto-accept or auto-correct labels from classifier output.

## Decision

Use option 2. Add `source_identity_confidence_proposal` as a transport-free
review layer that emits freshness-gated, operator-facing source identity review
records. The records include downstream rejection reasons and priority metadata,
but no queue import, no live apply, and no requested console change.

## Why this won

This keeps source identity confidence visible at the Product Layer boundary
before it can influence many processing families. It creates one contract for
operator review, acceptance, replay, and downstream hold reasons without
duplicating confidence gates across EQ, dynamics, FX, arrangement role, and
frequency separation code.

## Rejected alternatives

Raw classifier metadata was rejected because it does not explain whether
downstream proposals should be held. Per-module confidence gates were rejected
because they would drift and make source identity decisions harder to audit.
Auto-correction was rejected because a wrong source label can mislead multiple
mix decisions.

## Implementation plan

- Add source identity observation and alternative dataclasses.
- Normalize source type aliases and optional current-state freshness.
- Block stale provided freshness.
- Detect unknown, low-confidence, ambiguous, and unreliable-signal identities.
- Emit source identity review proposals with operator-acceptance and downstream
  rejection evidence.
- Add a preview adapter, registry method, proposal-only contract, and capability
  matrix reference.
- Add focused tests for fresh, stale, high-confidence, adapter, registry, and
  transport-free behavior.

## Test plan

```bash
PYTHONPATH=backend python -m py_compile \
  backend/source_identity_confidence_proposal.py \
  backend/processing_registry.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_source_identity_confidence_proposal.py
PYTHONPATH=backend python -m pytest \
  tests/test_source_identity_confidence_proposal.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Risks and rollback

Risk: early thresholds may hold too many labels until replay tuning exists.
Rollback is removing the review module, registry entry, proposal-only contract,
capability-row method id, and focused tests. No live transport path is added.
