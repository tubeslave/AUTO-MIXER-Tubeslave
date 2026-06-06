# Proposal: source-identity-calibration-memory-trust-promotion-plan-20260606

## Short Thesis

Add a read-only manual promotion plan for accepted source identity memory-trust
records, keeping actual downstream priority changes behind a later promotion
contract.

## What I Understood

The Automixer needs long-term memory to eventually improve source identity
trust. The chain now reaches operator acceptance, but acceptance still should
not update runtime trust. A manual plan can surface the next set of candidates
without granting authority.

## Proposed Solution

- Add `source_identity_calibration_memory_trust_promotion_plan.py`.
- Revalidate acceptance-ledger envelope, status, safety flags, summary counts,
  summary source-type lists, accepted records, and false-authority flags.
- Emit manual promotion candidates only from valid accepted records.
- Wire promotion-plan status and counts into `main_pipeline_preview`.
- Register the plan as `shadow_only` memory-learning evidence.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_trust_promotion_plan.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_memory_trust_promotion_plan.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-calibration-memory-trust-promotion-plan-20260606.md`

## Alternatives Considered

- Let accepted memory trust immediately update downstream priority. Rejected
  because promotion still needs separate replay freshness, rollback, and
  proposal-family behavior.
- Stop at acceptance ledger. Rejected because future promotion work needs a
  stable candidate payload.
- Make the plan queueable. Rejected because promotion planning is not a mixer
  action.

## Risks

- Manual promotion candidates may be mistaken for active trust authority.
  Mitigate by forcing all queue, model, runtime-memory, downstream-priority, and
  live flags false.
- Forged accepted records could claim authority. Mitigate by revalidating ledger
  and record false-authority fields before surfacing candidates.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_plan.py tests/test_source_identity_calibration_memory_trust_acceptance_ledger.py tests/test_source_identity_calibration_memory_trust_review.py tests/test_source_identity_calibration_memory_index.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_plan.py backend/source_identity_calibration_memory_trust_acceptance_ledger.py backend/source_identity_calibration_memory_trust_review.py backend/source_identity_calibration_memory_index.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue that a promotion plan is too cautious after operator
acceptance. I would keep it because downstream priority changes affect many
proposal families and still need a separate promotion contract with rollback.
