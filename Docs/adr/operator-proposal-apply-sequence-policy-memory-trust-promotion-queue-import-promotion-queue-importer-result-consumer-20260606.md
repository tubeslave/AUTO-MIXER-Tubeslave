# ADR: queue-importer result consumer for promotion queue import

## Context

The sequence-policy memory-trust promotion chain can now verify promotion queue-ordering-plan records and emit a queue-importer request template. The queue importer dry-runs by default and applies only to an explicit `OperatorProposalQueue` with `apply=true` and `operator_ack=true`.

## Options considered

1. Let dry-run importer results feed preflight planning.
2. Require applied queue-importer evidence before preflight planning.
3. Invoke the preflight planner from inside the result consumer.

## Decision

Use option 2. Add a read-only queue-importer result consumer that blocks dry-run importer output and emits only a preflight-plan request template after applied queue-importer evidence is verified.

## Why this won

The preflight planner operates on pending operator queue actions. Dry-run specs are review evidence, not queue state. Requiring applied import evidence preserves the operator-gated boundary while moving the chain toward an end-to-end supervised apply sequence.

## Rejected alternatives

- Dry-run-to-preflight was rejected because it could imply queue state that does not exist.
- Direct preflight invocation was rejected because result consumers should verify provenance and expose request templates, not run the next handler.

## Implementation plan

- Add the queue-importer result consumer module.
- Register it in Product Layer dispatch, processing registry, and capability matrix.
- Add focused tests for missing inputs, dry-run blocking, applied result availability, forged evidence rejection, and registry/matrix coverage.
- Add Product Layer dispatch coverage for the applied-import-to-preflight request path.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_promotion_queue_importer_result_consumer_feeds_preflight_plan_without_live_write -q`
- `python3 -m py_compile` for changed backend modules.
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: importer result schema changes can make the verifier too strict. Rollback is to remove the new consumer and dispatch registration, leaving the previous queue-importer request boundary intact.
