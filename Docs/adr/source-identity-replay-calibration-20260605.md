# ADR: source-identity-replay-calibration-20260605

## Context

The project goal requires long-term improvement from accumulated decisions,
tests, logs, and previous concerts. Source identity confidence now has an
operator review layer, but the system still needs replay-derived evidence before
accepted or rejected source labels can influence downstream proposal priority.

## Options considered

1. Keep replay labels as raw history only.
2. Add a proposal-only source identity replay calibration layer.
3. Let every downstream module build its own replay trust rules.
4. Automatically mutate classifier thresholds from accepted/rejected labels.

## Decision

Use option 2. Add `source_identity_replay_calibration` as a transport-free layer
that aggregates source identity replay outcomes into operator-review calibration
records. The layer can recommend downstream trust states, but it does not mutate
models, memory, queues, or live console state.

## Why this won

This creates a single auditable trust surface for source labels. It lets future
arrangement, EQ, dynamics, FX, frequency-separation, and balance proposals see
why a label is trusted or held without duplicating replay logic across modules.

## Rejected alternatives

Raw history was rejected because downstream families need machine-readable
trust states. Per-module replay rules were rejected because they would drift and
make source identity decisions harder to audit. Automatic threshold mutation was
rejected because calibration needs operator review, offline replay validation,
and rollback before it can affect runtime behavior.

## Implementation plan

- Add replay event and calibration recommendation dataclasses.
- Normalize source type aliases and optional current-state freshness.
- Block stale provided freshness.
- Aggregate accepted, corrected, rejected, replay-match, override, and safety
  regression counts by predicted source type.
- Emit proposal-only calibration records with non-queueable payloads and
  explicit no-model/no-memory mutation evidence.
- Register the method, proposal-only contract, and capability matrix row.
- Add focused tests for trusted, held, safety-blocked, stale, registry, and
  transport-free behavior.

## Test plan

```bash
PYTHONPATH=backend python -m py_compile \
  backend/source_identity_replay_calibration.py \
  backend/processing_registry.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_source_identity_replay_calibration.py
PYTHONPATH=backend python -m pytest \
  tests/test_source_identity_replay_calibration.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Risks and rollback

Risk: early calibration thresholds may hold labels too conservatively until more
show data exists. Rollback is removing the calibration module, registry entry,
proposal-only contract, capability-row method id, and focused tests. No live
transport, model mutation, or memory mutation path is added.
