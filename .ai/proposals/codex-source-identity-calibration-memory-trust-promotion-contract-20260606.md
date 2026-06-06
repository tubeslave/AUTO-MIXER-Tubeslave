# Proposal: source-identity-calibration-memory-trust-promotion-contract-20260606

## Short Thesis

Add a read-only promotion contract for accepted source identity memory-trust
records, requiring fresh replay context and rollback evidence before operator
review can consider any future downstream priority promotion.

## What I Understood

The Automixer is building a long-term memory loop for source identity trust. The
current chain reaches manual promotion candidates, but those candidates still
must not change runtime trust or proposal priority. A contract layer can make the
next evidence gate explicit without adding an executor.

## Proposed Solution

- Add `source_identity_calibration_memory_trust_promotion_contract.py`.
- Validate the promotion-plan envelope, summary counts, candidate shape, and
  false-authority flags.
- Require fresh read-only replay context.
- Require read-only rollback evidence with a manual revert strategy.
- Wire contract status and record counts into `main_pipeline_preview`.
- Register the contract as `shadow_only` memory-learning evidence.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_trust_promotion_contract.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_memory_trust_promotion_contract.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-calibration-memory-trust-promotion-contract-20260606.md`

## Alternatives Considered

- Let promotion-plan candidates immediately update downstream priority. Rejected
  because the project still lacks a reviewed executor and rollback path.
- Put rollback requirements directly into the promotion plan. Rejected because
  candidate extraction and promotion readiness are different contracts.
- Make the contract queueable. Rejected because memory-trust promotion is not a
  mixer action and should not enter the live decision queue.

## Risks

- Operators may mistake contract-ready records for actual trust promotion.
  Mitigate by forcing all queue, model, runtime-memory, downstream-priority, and
  live authority flags false.
- Forged rollback evidence could imply safety. Mitigate by validating rollback
  safety flags and allowed strategy before surfacing contract records.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_contract.py tests/test_source_identity_calibration_memory_trust_promotion_plan.py tests/test_source_identity_calibration_memory_trust_acceptance_ledger.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_contract.py backend/source_identity_calibration_memory_trust_promotion_plan.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue the contract is still too cautious and should execute the
promotion. I would keep it review-only because downstream priority changes can
affect multiple proposal families and still need an executor with rollback.
