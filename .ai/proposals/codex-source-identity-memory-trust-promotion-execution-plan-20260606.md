# Proposal: source-identity-memory-trust-promotion-execution-plan-20260606

## Short Thesis

Add a read-only execution plan for accepted memory-trust promotion records, but
keep actual trust promotion and downstream priority mutation behind a future
executor contract.

## What I Understood

The project is building a long-term source identity learning loop. Promotion
acceptance proves an operator approved a contract record, but that approval must
not automatically change proposal priority, runtime memory, models, queues, or
live mixer state.

## Proposed Solution

- Add `source_identity_calibration_memory_trust_promotion_execution_plan.py`.
- Validate the promotion acceptance ledger envelope, status, safety flags,
  summary counts, accepted records, and false-authority fields.
- Convert accepted records into non-executing execution candidates.
- Wire plan status and counts into `main_pipeline_preview`.
- Register the plan as `shadow_only` memory-learning evidence.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_trust_promotion_execution_plan.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_memory_trust_promotion_execution_plan.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-memory-trust-promotion-execution-plan-20260606.md`

## Alternatives Considered

- Let accepted promotion records update downstream trust immediately. Rejected
  because executor semantics, rollback application, and safety promotion are
  still separate.
- Extend the acceptance ledger to also be the execution plan. Rejected because
  it blurs operator decision evidence with future executor planning.
- Make execution candidates queueable. Rejected because this is not a mixer
  action and must not mutate the Product Layer queue.

## Risks

- The plan could be mistaken for executor authority. Mitigate by forcing all
  execution, queue, model, runtime-memory, downstream-priority, and live
  authority flags false.
- Forged ledgers could claim unsafe authority. Mitigate by rejecting unsafe
  fields, mismatched counts, invalid accepted records, and duplicate contract
  review decisions.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_execution_plan.py tests/test_source_identity_calibration_memory_trust_promotion_acceptance_ledger.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_execution_plan.py backend/source_identity_calibration_memory_trust_promotion_acceptance_ledger.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue the name "execution plan" implies too much. I would keep it
because the explicit false authority flags and registry status make the missing
executor boundary visible instead of hiding the next required step.
