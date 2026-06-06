# ADR: source-identity-calibration-memory-trust-acceptance-ledger-20260606

## Context

Source identity calibration memory can now be written, indexed, and surfaced as
operator trust-review candidates with fresh replay context. The next boundary
is recording operator accepted/rejected decisions on those candidates without
changing runtime behavior.

## Options Considered

1. Let accepted memory-trust review candidates immediately update downstream
   source identity trust.
2. Keep review candidates without an acceptance ledger.
3. Add a separate read-only memory-trust acceptance ledger.

## Decision

Use option 3. Add a transport-free ledger that validates memory-trust review
candidates and operator decisions, then records accepted/rejected outcomes as
evidence only.

## Why This Won

This makes the learning chain auditable while keeping promotion separate. The
system can know which memory-trust candidates an operator accepted, but models,
runtime memory, downstream proposal priority, queues, and live console state
remain unchanged.

## Rejected Alternatives

- Direct trust promotion was rejected because acceptance alone does not define
  replay freshness, rollback, or proposal-family behavior.
- No ledger was rejected because future promotion needs a durable operator
  decision surface.
- Queueable acceptance was rejected because memory-trust acceptance is not an
  executable mixer action.

## Implementation Plan

1. Add a backend acceptance-ledger module with strict trust-review validation.
2. Validate review envelope, status, safety flags, summary counts, candidate
   fields, decision records, UTC timestamps, and unsafe authority flags.
3. Wire ledger status and counts into `main_pipeline_preview`.
4. Register the ledger as `shadow_only` memory-learning evidence.
5. Add focused ledger, Product Layer, registry, and capability tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_acceptance_ledger.py tests/test_source_identity_calibration_memory_trust_review.py tests/test_source_identity_calibration_memory_index.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_acceptance_ledger.py backend/source_identity_calibration_memory_trust_review.py backend/source_identity_calibration_memory_index.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: accepted records could be mistaken for downstream priority promotion.
Mitigation: ledger records force
`memory_trust_promotion_allowed_by_acceptance=false`,
`downstream_priority_change_allowed_by_acceptance=false`,
`queue_mutation_allowed_by_acceptance=false`,
`model_mutation_allowed_by_acceptance=false`,
`runtime_memory_mutation_allowed_by_acceptance=false`,
`live_authority_granted=false`, and `full_autonomous_ready=false`.

Rollback: remove the ledger call from `processing_pipeline_preview.py`, remove
the registry/capability rows, and keep memory-trust review candidates as
operator-review evidence only.
