# Brief: operator proposal apply-sequence policy memory trust promotion queue-import promotion execution-result consumer

## Context

The promotion execution-gate result consumer can verify generic gate results and expose a generic sequence-executor request template. The generic executor may call `apply_operator_proposal` only after separate `execute=true`, `operator_ack=true`, and `approved=true` inputs; its result is execution evidence, not audit evidence.

## Task

Add a fail-closed verifier that consumes a verified promotion execution-gate result consumer plus a generic sequence-executor result. It should expose only an `audit_operator_proposal_apply_sequence_execution` request template after every execution row is applied and validated.

## Safety constraints

- Do not call mixer transport or manual console write paths from the verifier.
- Do not invoke the execution-audit handler inside the verifier.
- Do not reserve idempotency, mutate proposal status or order, mutate queues, mutate policy/runtime/model/memory state, or grant live authority.
- Accept delegated `apply_operator_proposal` evidence only when the generic execution-result consumer validates it against the generic gate provenance.
- Block partial/blocked executor results, forged live-write authority, direct manual console-write claims, idempotency mismatch, unsafe fader values, or mismatched promotion provenance.

## Validation

- Focused unit tests for missing inputs, applied execution evidence, blocked rows, forged evidence rejection, registry, and capability matrix.
- Product Layer dispatch test for promotion execution-gate result -> execution-result verifier -> execution-audit request template.
- Compile, memory lint, diff check, and focused pytest.
