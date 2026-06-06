# ADR: batch-preflight result consumer for promotion queue import

## Context

The promotion queue-import path can now verify queue-import preflight plans and expose a batch-preflight request template. The batch-preflight handler reports ready/blocked proposal rows without mutating the queue, reserving idempotency, or writing to the mixer.

## Options considered

1. Feed batch-preflight results directly into generic sequence planning.
2. Add a promotion batch-preflight result consumer that verifies provenance before exposing a sequence-plan request template.
3. Invoke the generic sequence-plan handler from inside the consumer.

## Decision

Use option 2. Add a read-only promotion batch-preflight result consumer that verifies the prior promotion preflight-plan consumer, delegates batch-result validation to the existing generic queue-import preflight-result consumer, and emits only a sequence-plan request template.

## Why this won

It keeps the promotion chain explicit without duplicating the existing batch-preflight validation contract. It also preserves the Product Layer boundary: the consumer verifies evidence, but the caller must separately invoke sequence planning.

## Rejected alternatives

- Direct batch-to-sequence planning was rejected because it bypasses promotion provenance checks.
- Direct handler invocation was rejected because it would combine verification with execution of the next planning step.
- Duplicating generic validation was rejected because it increases drift risk for preflight readiness rules.

## Implementation plan

- Add the promotion batch-preflight result consumer module.
- Register it in Product Layer dispatch, processing registry, and capability matrix.
- Add focused tests for missing inputs, ready evidence, blocked rows, forged evidence, registry, and matrix coverage.
- Extend Product Layer dispatch coverage through generic sequence planning while preserving no live write and no idempotency reservation.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_promotion_queue_importer_result_consumer_feeds_preflight_plan_without_live_write tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py -q`
- `python3 -m py_compile` for changed backend modules.
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: the wrapper rejects valid future generic batch-preflight schema changes. Rollback is to remove the new consumer and dispatch registration, leaving the promotion preflight-plan result consumer as the final promotion boundary before generic batch preflight.
