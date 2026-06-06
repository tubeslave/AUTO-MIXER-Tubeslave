# Proposal: dry-run result consumer for promotion queue import

## Short thesis

Add a read-only result consumer between the promotion dry-run request and the generic execution-gate handler. It should verify promotion provenance and dry-run `would_apply` rows, then emit only an execution-gate request template.

## What I understood

The promotion chain can now request and run a generic dry-run. That dry-run is rehearsal evidence, not execution authority. A separate promotion verifier should ensure the dry-run matches the prior promotion sequence-plan evidence before any execution-gate request is exposed.

## Proposed solution

- Add `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer`.
- Verify the prior promotion sequence-plan result consumer envelope, safety flags, planned ids, and emitted dry-run request.
- Reuse the existing generic queue-import dry-run result consumer to validate the dry-run payload itself.
- Reject blocked rows, forged live-write flags, idempotency mismatch, unsafe fader values, and mismatched provenance.
- Emit only a `gate_operator_proposal_apply_sequence_execution` request template with `operator_ack=false`.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`

## Alternatives considered

- Let the sequence-plan result consumer emit an execution-gate request directly. Rejected because dry-run evidence must be verified first.
- Duplicate generic dry-run validation. Rejected because the existing queue-import dry-run result consumer owns that contract.
- Invoke the execution gate from inside this consumer. Rejected because result consumers should expose request templates, not execute the next Product Layer step.

## Risks

- The wrapper depends on the generic dry-run result schema. Mitigation: invalid generic evidence is surfaced under `generic_dry_run_result_consumer:*` invalid ids and focused tests cover forged paths.
- The gate request can look close to live execution. Mitigation: it keeps `operator_ack=false`; later gate/executor/audit/supervised-apply stages remain separate.

## Test plan

- Focused pytest for the new consumer.
- Product Layer dispatch pytest through execution-gate request exposure.
- Registry and capability matrix checks.
- `py_compile`, `memory_lint`, and `git diff --check`.

## Where the other agent may disagree

Kimi might prefer the promotion verifier to re-parse every dry-run row itself. I would delegate row-level validation to the generic verifier to avoid split safety rules.
