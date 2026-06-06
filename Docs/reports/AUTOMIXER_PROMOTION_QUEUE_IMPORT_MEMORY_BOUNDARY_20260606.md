# Automixer Promotion Queue Import Memory Boundary - 2026-06-06

## Purpose

This note maps the current promotion queue-import memory boundary so operators
and future agents do not confuse request templates with executed evidence or
memory writes.

The three newest promotion consumers are read-only verifiers. They may expose
the next request payload shape only after matching generic evidence back to
promotion provenance. They do not invoke downstream handlers, write memory
files, mutate Product Layer state, reserve idempotency keys, or grant live
authority.

## Boundary Map

| Step | Input Evidence | Output | What It Does Not Do |
| --- | --- | --- | --- |
| Promotion audit-policy-gate result consumer | Generic audit-policy-gate result plus promotion execution-audit provenance | Memory-draft request template | Does not call memory draft handler, mutate policy, mutate memory, reserve idempotency, or touch mixer transport |
| Promotion memory-draft result consumer | Generic policy-memory draft plus promotion audit-policy-gate provenance | Memory-writer request template with `apply=false` and `operator_ack=false` | Does not call writer, write `.ai/memory/project.md`, mutate runtime memory, mutate policy, or touch mixer transport |
| Promotion memory-writer dry-run result consumer | Generic memory-writer dry-run plus promotion memory-draft provenance | Memory-writer apply request template with `apply=false` and `operator_ack=false` | Does not apply writer output, write memory, mutate queues, mutate proposals, or touch mixer transport |

## Template Versus Execution

Request templates are review artifacts. They are not proof that a downstream
handler ran, and they are not approval to run one.

The current safety contract keeps these flags false at the consumer boundary:

- `memory_draft_handler_invoked`
- `memory_writer_handler_invoked`
- `memory_file_write_applied`
- `memory_file_mutation`
- `memory_mutation`
- `policy_mutation`
- `runtime_memory_mutation`
- `model_mutation`
- `queue_mutation`
- `proposal_status_mutation`
- `proposal_order_mutation`
- `idempotency_reserved`
- `manual_console_write_called`
- `live_write`
- `live_authority_granted`
- `full_autonomous_ready`

Any future cycle that turns one of these templates into an executed writer or
promotion action must be a separate supervised task with explicit operator
acknowledgement, focused tests, and new audit evidence.

## Current Validation Evidence

Focused validation for this boundary:

```bash
PYTHONPATH=backend python3 -m pytest -q \
  tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer.py \
  tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer.py \
  tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer.py
```

Expected result in the 2026-06-06 validation cycle: `21 passed`.

```bash
PYTHONPATH=backend python3 -m pytest -q tests/test_operator_product_state.py
```

Expected result in the 2026-06-06 validation cycle: `142 passed`.

## Live Safety

This boundary does not perform live OSC/WING writes. It does not connect to a
real console and does not call any forbidden mixer command such as fader, gain,
trim, EQ, compressor, routing, mute, scene, snapshot, insert, or FX writes.
