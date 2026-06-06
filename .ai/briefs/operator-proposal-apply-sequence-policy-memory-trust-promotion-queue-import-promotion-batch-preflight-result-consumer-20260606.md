# Brief: operator proposal apply-sequence policy memory trust promotion queue-import promotion batch-preflight result consumer

## Context

The promotion preflight-plan result consumer can verify queue-import preflight plans and emit a batch-preflight request template. The existing batch preflight handler remains read-only and reports whether imported proposals are ready for the generic sequence-plan handler.

## Task

Add a fail-closed verifier that consumes a verified promotion preflight-plan result consumer plus an approved batch-preflight result. It should expose only a `plan_operator_proposal_apply_sequence` request template after every batch preflight row is ready and matches promotion provenance.

## Safety constraints

- Do not call mixer transport or manual console write paths.
- Do not run the sequence-plan handler inside the verifier.
- Do not reserve idempotency, mutate proposal status or order, mutate queues, mutate policy/runtime/model/memory state, or grant live authority.
- Block batch-preflight results with blocked rows, mismatched order, forged safety flags, or missing approval.
- Real live writes must still flow through the existing supervised `apply_operator_proposal` path after later gates.

## Validation

- Focused unit tests for missing inputs, ready batch evidence, blocked rows, forged evidence rejection, registry, and capability matrix.
- Product Layer dispatch test for promotion preflight-plan result -> batch preflight -> verifier -> generic sequence-plan request.
- Compile, memory lint, diff check, and focused pytest.
