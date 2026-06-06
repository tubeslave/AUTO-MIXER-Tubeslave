# Brief: operator proposal apply-sequence policy memory trust promotion queue-import promotion queue-importer result consumer

## Context

The promotion queue-ordering-plan result consumer now emits a queue-importer request template. The importer can dry-run by default and can mutate only an explicit `OperatorProposalQueue` when called with `apply=true` and `operator_ack=true`.

## Task

Add the next fail-closed verifier that consumes a verified promotion queue-ordering-plan result consumer plus a queue-importer result. It should expose only a queue-import preflight-plan request template after an applied queue-importer result is present.

## Safety constraints

- Do not call mixer transport or manual console write paths.
- Do not import queues inside the verifier.
- Do not run preflight, reserve idempotency, mutate proposal status, mutate policy/runtime/model/memory state, or grant live authority.
- Dry-run importer evidence must remain blocked before preflight planning.

## Validation

- Focused unit tests for missing inputs, dry-run blocking, applied-result request availability, forged authority rejection, registry, and capability matrix.
- Product Layer dispatch test for applied queue import -> verifier -> preflight plan request.
- Compile, memory lint, diff check, and focused pytest.
