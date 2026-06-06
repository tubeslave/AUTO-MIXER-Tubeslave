# Brief: operator proposal apply-sequence policy memory trust promotion queue-import promotion dry-run result consumer

## Context

The promotion sequence-plan result consumer can verify generic sequence plans and emit a dry-run request template. The existing generic dry-run handler rehearses the planned apply sequence without calling `apply_operator_proposal`, reserving idempotency, mutating queue state, or touching live mixer transport.

## Task

Add a fail-closed verifier that consumes a verified promotion sequence-plan result consumer plus a generic dry-run result. It should expose only a `gate_operator_proposal_apply_sequence_execution` request template after every dry-run row is `would_apply`.

## Safety constraints

- Do not call mixer transport or manual console write paths.
- Do not run the execution-gate handler inside the verifier.
- Do not reserve idempotency, mutate proposal status or order, mutate queues, mutate policy/runtime/model/memory state, or grant live authority.
- Keep `operator_ack=false` in the emitted execution-gate request; a separate operator action must set acknowledgement before the generic gate can mark rows ready.
- Block dry-runs with blocked rows, forged live-write authority, idempotency mismatch, unsafe fader values, or mismatched promotion provenance.

## Validation

- Focused unit tests for missing inputs, ready dry-run evidence, blocked rows, forged evidence rejection, registry, and capability matrix.
- Product Layer dispatch test for promotion sequence-plan result -> generic dry-run -> verifier -> execution-gate request template.
- Compile, memory lint, diff check, and focused pytest.
