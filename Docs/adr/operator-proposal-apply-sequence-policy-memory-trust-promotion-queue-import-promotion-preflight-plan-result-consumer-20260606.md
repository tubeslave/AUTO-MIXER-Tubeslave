# ADR: preflight-plan result consumer for promotion queue import

## Context

The sequence-policy memory-trust promotion chain can verify applied queue-importer results and request a read-only queue-import preflight plan. The plan selects imported pending proposals and prepares a request for the existing `preflight_operator_proposal_apply_batch` handler, but it is not authority to run preflight or apply anything.

## Options considered

1. Let the queue-importer result consumer emit a batch-preflight request directly.
2. Add a preflight-plan result consumer that verifies plan provenance before exposing a batch-preflight request template.
3. Invoke the batch preflight handler from inside the result consumer.

## Decision

Use option 2. Add a read-only preflight-plan result consumer that validates the queue-import preflight plan against applied queue-importer provenance and emits only a batch-preflight request template.

## Why this won

The queue-import preflight planner owns candidate selection from current pending queue actions. A separate consumer can verify that evidence before the chain reaches batch preflight, while preserving the existing Product Layer handler boundaries and keeping approval explicit.

## Rejected alternatives

- Direct queue-importer-to-batch-preflight was rejected because it skips the dedicated queue-import preflight plan evidence layer.
- Direct batch preflight invocation was rejected because result consumers must not execute the next handler or reserve idempotency.
- Pre-approving the emitted request was rejected because a human/operator-controlled request must set `approved=true` at the batch preflight boundary.

## Implementation plan

- Add the preflight-plan result consumer module.
- Register it in Product Layer dispatch, processing registry, and capability matrix.
- Add focused tests for missing inputs, dry-run importer blocking, ready plan availability, forged plan rejection, and registry/matrix coverage.
- Extend Product Layer dispatch coverage through batch preflight while preserving no live write and no idempotency reservation.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer.py tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_promotion_queue_importer_result_consumer_feeds_preflight_plan_without_live_write tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py -q`
- `python3 -m py_compile` for changed backend modules.
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: the verifier can reject valid preflight plans after schema changes. Rollback is to remove the new consumer and dispatch registration, leaving the applied queue-importer result consumer as the final promotion boundary before generic preflight planning.
