# Proposal: operator-proposal-apply-sequence-policy-memory-trust-promotion-apply-audit-20260606

## Short thesis

Applied sequence-policy promotion results need a read-only rollback audit
before any policy consumer can treat them as trusted evidence.

## What I understood

The executor can now write to an explicit caller-provided policy store. That is
useful progress toward learning, but it still needs audit evidence showing
what changed, what rollback state is required, and which authorities remain
blocked.

## Proposed solution

Add `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit.py`.
It consumes the executor output, validates applied results, produces rollback
audit records, and rejects forged live, queue, proposal-state, or sequence
execution authority.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/processing_pipeline_preview.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-apply-audit-20260606.md`

## Alternatives considered

- Let executor apply results feed policy consumers directly. Rejected: applied
  results still need rollback evidence and independent audit.
- Merge audit into the executor. Rejected: audit should be a separate
  trust-boundary layer.

## Risks

- Forged apply results could claim sequence execution or live authority. The
  audit rejects unsafe result flags and inconsistent executor summaries.
- A rollback gap could be hidden. The audit records rollback target state and
  keeps `rollback_verified=false`.

## Test plan

- Focused audit tests.
- Product Layer preview test.
- Registry/capability matrix tests.
- Compile, memory lint, diff check, and full backend suite.

## Where the other agent may disagree

Kimi may suggest moving straight to rollback verification or policy consumers.
The audit layer is needed first so rollback verification has a stable input
contract and policy consumers cannot bypass review.
