# Brief: operator proposal apply-sequence policy memory trust promotion queue-import promotion execution-gate result consumer

## Context

The promotion dry-run result consumer can verify generic dry-run results and emit a generic execution-gate request template. The generic execution gate can mark rows ready for a separate executor, but it still does not call `apply_operator_proposal`, reserve idempotency, mutate queues, or touch mixer transport.

## Task

Add a fail-closed verifier that consumes a verified promotion dry-run result consumer plus a generic execution-gate result. It should expose only an `execute_operator_proposal_apply_sequence` request template after every gate row is ready for executor.

## Safety constraints

- Do not call mixer transport or manual console write paths.
- Do not invoke the sequence executor inside the verifier.
- Do not reserve idempotency, mutate proposal status or order, mutate queues, mutate policy/runtime/model/memory state, or grant live authority.
- Keep `execute=false`, `operator_ack=false`, and `approved=false` in the emitted executor request; a separate operator action must set all three before executor use.
- Block gate results with blocked rows, forged live-write authority, idempotency mismatch, unsafe fader values, or mismatched promotion provenance.

## Validation

- Focused unit tests for missing inputs, ready gate evidence, blocked rows, forged evidence rejection, registry, and capability matrix.
- Product Layer dispatch test for promotion dry-run result -> generic execution gate -> verifier -> executor request template.
- Compile, memory lint, diff check, focused pytest, and full pytest before commit.
