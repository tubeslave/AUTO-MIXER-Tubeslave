# Proposal: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-memory-draft-result-consumer-20260606

## Short thesis

Add a read-only result consumer between generic policy-memory draft output and
generic memory-writer input, preserving sequence-policy queue-import provenance
without writing memory or granting live authority.

## What I understood

The queue-import path reuses generic apply-sequence and memory-learning handlers
while requiring explicit verification at every boundary. The memory draft result
is evidence, not permission to append memory. It needs a queue-import-specific
consumer before exposing the writer request.

## Proposed solution

- Build
  `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer`.
- Require the upstream audit-policy-gate result consumer to be available and
  read-only.
- Require a generic `operator_proposal_apply_sequence_policy_memory_draft` with
  ready records, append-only `.ai/memory/project.md` target, verified evidence,
  rollback, fader/gain bounds, matching idempotency, and false mutation flags.
- Emit only a `write_operator_proposal_apply_sequence_policy_memory` request
  template with `apply=false` and `operator_ack=false`.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-memory-draft-result-consumer-20260606.md`

## Alternatives considered

Directly passing memory drafts to the writer is faster but drops queue-import
provenance checks. Automatically invoking the writer is unsafe because the
writer is the first boundary that can mutate a memory file.

## Risks

The main risk is that a writer request template is mistaken for permission to
write memory. The request must keep `apply=false`, `operator_ack=false`, writer
invocation false, memory-file mutation false, live authority false, and full
autonomous readiness false.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer.py backend/handlers/product_state_handlers.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_execution_result_consumer_feeds_audit_without_extra_write tests/test_project_goal_capability_matrix.py -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may argue for running the writer dry-run immediately. Keeping this layer as
a request-template producer is cleaner: the generic writer remains its own
auditable boundary and the consumer never mutates memory.
