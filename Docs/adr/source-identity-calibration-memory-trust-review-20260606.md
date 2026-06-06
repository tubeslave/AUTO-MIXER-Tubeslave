# ADR: source-identity-calibration-memory-trust-review-20260606

## Context

Source identity calibration has an append-only memory writer and read-only
memory index. The remaining gap before any future downstream trust promotion is
an operator-review contract that checks indexed memory against fresh replay
context without changing runtime behavior.

## Options Considered

1. Let indexed memory directly influence downstream source identity trust.
2. Keep indexed memory as raw evidence only.
3. Add a separate memory trust-review layer with fresh-context gating.

## Decision

Use option 3. Add a transport-free trust-review builder that revalidates memory
index evidence, requires fresh source identity replay context, and emits
operator-review candidates only.

## Why This Won

This keeps the long-term learning path moving while preserving the safety
boundary. The system can see which memory records are ready for human review,
but models, runtime memory, downstream proposal priority, queues, and live
console state remain unchanged.

## Rejected Alternatives

- Direct downstream trust influence was rejected because current source context,
  replay freshness, and operator promotion are still separate proof gates.
- Raw index-only evidence was rejected because future promotion work needs a
  stable review-candidate payload.
- Queueable trust review was rejected because trust review is not an executable
  mixer action.

## Implementation Plan

1. Add a backend trust-review module with strict index validation.
2. Validate index envelope, status, safety flags, summary counts, summary
   identity lists, record fields, UTC timestamps, and false-authority flags.
3. Require fresh source identity replay context before surfacing candidates.
4. Wire trust-review status into `main_pipeline_preview`.
5. Register trust review as `shadow_only` memory-learning evidence.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_review.py tests/test_source_identity_calibration_memory_index.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_review.py backend/source_identity_calibration_memory_index.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: review candidates could be mistaken for downstream-priority authority.
Mitigation: review candidates force `queue_mutation_allowed_by_review=false`,
`model_mutation_allowed_by_review=false`,
`runtime_memory_mutation_allowed_by_review=false`,
`downstream_priority_change_allowed_by_review=false`,
`live_authority_granted=false`, and `full_autonomous_ready=false`.

Rollback: remove the trust-review call from `processing_pipeline_preview.py`,
remove registry/capability rows, and keep indexed source identity memory as
read-only evidence until a replacement review contract is agreed.
