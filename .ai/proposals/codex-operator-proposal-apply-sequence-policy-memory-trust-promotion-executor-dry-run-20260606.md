# Proposal: operator-proposal-apply-sequence-policy-memory-trust-promotion-executor-dry-run-20260606

## Short thesis

Accepted sequence-policy promotion execution candidates should be rehearsed as
deterministic dry-run specs before any executor can mutate the policy gate.

## What I understood

The project is building a controlled path from operator-reviewed supervised
apply-sequence memory toward future policy learning. The execution plan
identifies candidate records, but the next boundary should still be read-only:
show exactly what a later executor would apply, with stable idempotency and
rollback fields.

## Proposed solution

Add `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run.py`.
It validates the promotion execution plan, rejects unsafe or ambiguous
candidates, emits dry-run records targeting `operator_sequence_policy_trust_gate`,
and blocks `apply=true`. The Product Layer preview will call it with
`apply=false` and `operator_ack=false`.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/processing_pipeline_preview.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-executor-dry-run-20260606.md`

## Alternatives considered

- Let the execution plan double as the dry-run. Rejected: the dry-run needs
  target paths, operations, idempotency keys, and apply blocking semantics.
- Build the applying executor immediately. Rejected: that would combine
  rehearsal and mutation in one trust-boundary change.

## Risks

- A forged execution plan could claim policy or runtime-memory authority. The
  dry-run rejects unsafe envelope, summary, and candidate-level claims.
- Duplicate candidates could produce conflicting specs. The dry-run rejects
  duplicate policy idempotency keys before emitting records.

## Test plan

- Focused dry-run tests.
- Product Layer preview test.
- Registry/capability matrix tests.
- Compile, memory lint, diff check, and full backend suite.

## Where the other agent may disagree

Kimi may argue for adding the applying executor now. The safer step is to land
the dry-run contract first so the applying executor has a precise input and
review surface.
