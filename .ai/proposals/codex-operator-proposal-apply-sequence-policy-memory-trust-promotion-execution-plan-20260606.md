# Proposal: operator-proposal-apply-sequence-policy-memory-trust-promotion-execution-plan-20260606

## Short thesis

Accepted sequence-memory trust promotion records should become non-executing
future executor candidates, not sequence-policy or live apply authority.

## What I understood

The project is building an auditable route from supervised apply-sequence
memory into future policy learning. Promotion acceptance records are explicit
operator evidence, but they still need a read-only execution plan before any
separate executor can be designed safely.

## Proposed solution

Add `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan.py`.
It consumes the promotion acceptance ledger, validates ledger and accepted
record safety, refuses unreviewed or ambiguous contract state, and emits
non-executing candidates. Execution, queue, policy, model, runtime-memory, and
live-authority flags remain false.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/processing_pipeline_preview.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-execution-plan-20260606.md`

## Alternatives considered

- Treat accepted promotion records as direct executor authority. Rejected:
  acceptance is evidence, not apply permission.
- Wait until a full executor exists. Rejected: a separate execution plan gives
  a safer contract boundary and lets tests lock the no-authority semantics.

## Risks

- A forged acceptance ledger could claim policy or execution authority. The
  plan rejects unsafe envelope, summary, and record-level authority claims.
- Operators could mistake candidates for executable steps. The plan status,
  safety flags, and registry metadata state that a separate executor is still
  required.

## Test plan

- Focused execution-plan tests.
- Product Layer preview test.
- Registry/capability matrix tests.
- Compile, memory lint, diff check, and full backend suite.

## Where the other agent may disagree

Kimi may push for a combined executor. The safer order is to land the
read-only candidate contract first, then review an executor against a stable
input shape.
