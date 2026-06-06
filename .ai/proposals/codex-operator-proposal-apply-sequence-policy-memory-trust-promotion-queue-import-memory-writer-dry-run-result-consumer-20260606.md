# Proposal: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-memory-writer-dry-run-result-consumer-20260606

## Short thesis

Add a read-only result consumer between generic memory-writer dry-run output and
any append-only memory write request, preserving sequence-policy queue-import
provenance without writing memory or granting live authority.

## What I understood

The memory writer is the first generic handler in this chain that can mutate a
repo file when `apply=true` and `operator_ack=true`. Before any apply template is
shown, the dry-run result must prove the writer validated the draft records and
stopped only because apply and acknowledgement were absent.

## Proposed solution

- Build
  `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer`.
- Require the upstream memory-draft result consumer to be available and
  read-only.
- Require a generic memory-writer payload with
  `status=append_only_memory_write_dry_run_ready`, `success=false`, no
  write results, false memory-file mutation, and the expected dry-run blocker.
- Emit only a writer apply request template with `apply=false` and
  `operator_ack=false`.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-memory-writer-dry-run-result-consumer-20260606.md`

## Alternatives considered

Directly surfacing the generic writer apply request after memory draft is
simpler, but it skips proof that the writer accepted the draft in dry-run mode.
Automatically applying the writer is rejected because it mutates project memory.

## Risks

The main risk is that the apply request template is mistaken for permission to
write. The consumer keeps `apply=false`, `operator_ack=false`, writer invocation
false, memory-file mutation false, live authority false, and full autonomous
readiness false.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer.py backend/handlers/product_state_handlers.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_execution_result_consumer_feeds_audit_without_extra_write tests/test_project_goal_capability_matrix.py -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may argue the dry-run consumer is an extra ceremony. The counterpoint is
that the next boundary can append to `.ai/memory/project.md`, so dry-run proof is
worth the extra explicit check.
