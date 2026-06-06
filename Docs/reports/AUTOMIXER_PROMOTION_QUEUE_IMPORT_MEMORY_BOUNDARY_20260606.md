# Automixer Promotion Queue Import Memory Boundary - 2026-06-06

## Purpose

This note maps the current promotion queue-import memory boundary so operators
and future agents do not confuse request templates with executed evidence or
memory writes.

The six newest promotion consumers are read-only verifiers. They may expose
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
| Promotion memory-writer apply result consumer | Explicit generic append-only writer result plus promotion dry-run provenance | Memory-index request template | Does not index memory, perform a consumer-side memory write, mutate policy/runtime/model state, reserve idempotency, or touch mixer transport |
| Promotion memory-index result consumer | Read-only memory index result plus promotion writer-apply provenance and fresh sequence-review context | Memory-trust-review request template | Does not run trust review, mutate memory/policy/runtime/model state, mutate queues or proposals, reserve idempotency, or touch mixer transport |
| Promotion memory-trust-review result consumer | Memory trust-review result plus promotion memory-index provenance | Acceptance-ledger request template with empty `acceptance_records` | Does not run acceptance ledger, accept or reject candidates, mutate memory/policy/runtime/model state, mutate queues or proposals, mutate sequence-policy trust stores, reserve idempotency, or touch mixer transport |

## Template Versus Execution

Request templates are review artifacts. They are not proof that a downstream
handler ran, and they are not approval to run one.

The current safety contract keeps these flags false at the consumer boundary:

- `memory_draft_handler_invoked`
- `memory_writer_handler_invoked`
- `memory_index_handler_invoked`
- `memory_index_created`
- `memory_trust_review_handler_invoked`
- `memory_trust_review_created`
- `acceptance_ledger_handler_invoked`
- `acceptance_ledger_created`
- `memory_file_write_applied`
- `memory_file_write_applied_by_consumer`
- `memory_file_mutation`
- `memory_mutation`
- `policy_mutation`
- `runtime_memory_mutation`
- `model_mutation`
- `queue_mutation`
- `proposal_status_mutation`
- `proposal_order_mutation`
- `sequence_policy_trust_store_mutation`
- `proposal_state_mutation`
- `idempotency_reserved`
- `manual_console_write_called`
- `live_write`
- `live_authority_granted`
- `full_autonomous_ready`

Any future cycle that turns one of these templates into an executed writer or
promotion action must be a separate supervised task with explicit operator
acknowledgement, focused tests, and new audit evidence.

The one intentional exception in this boundary is source evidence from the
generic writer result: a promotion memory-writer apply consumer may report
`source_memory_file_write_applied=true` only when the upstream writer result
already proves an explicit append-only memory write with `apply=true` and
`operator_ack=true`. The promotion consumer itself still reports
`memory_file_write_applied_by_consumer=false` and `memory_file_mutation=false`.

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

Additional focused validation for the current writer-apply and memory-index
handoffs plus the current memory-trust-review acceptance request handoff:

```bash
PYTHONPATH=backend python3 -m pytest -q \
  tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer.py \
  tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer.py \
  tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer.py \
  tests/test_operator_product_state.py::test_sequence_policy_queue_import_promotion_queue_importer_result_consumer_feeds_preflight_plan_without_live_write
```

Observed result in the 2026-06-06 validation cycle: `24 passed`.

## Live Safety

This boundary does not perform live OSC/WING writes. It does not connect to a
real console and does not call any forbidden mixer command such as fader, gain,
trim, EQ, compressor, routing, mute, scene, snapshot, insert, or FX writes.
