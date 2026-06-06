# Brief: operator proposal apply-sequence policy memory trust promotion queue-import promotion execution-audit result consumer

## Context

The promotion execution-result consumer can verify generic sequence-executor results and expose a generic execution-audit request template. The generic execution audit can inspect delegated `apply_operator_proposal` evidence without applying anything, but its result still needs promotion provenance verification before any audit-policy gate request is exposed.

## Task

Add a fail-closed verifier that consumes a verified promotion execution-result consumer plus a generic execution-audit result. It should expose only a `gate_operator_proposal_apply_sequence_audit_policy` request template after every audit row passes.

## Safety constraints

- Do not call mixer transport or manual console write paths from the verifier.
- Do not invoke the audit-policy gate inside the verifier.
- Do not reserve idempotency, mutate proposal status or order, mutate queues, mutate policy/runtime/model/memory state, or grant live authority.
- Accept audit evidence only through the generic execution-audit result consumer.
- Block failed/blocked audit rows, forged live authority, widened write kinds, idempotency mismatch, unsafe fader values, or mismatched promotion provenance.

## Validation

- Focused unit tests for missing inputs, passed audit evidence, blocked rows, forged evidence rejection, registry, and capability matrix.
- Product Layer dispatch test for promotion execution-result -> execution-audit verifier -> audit-policy-gate request template.
- Compile, memory lint, diff check, and focused pytest.
