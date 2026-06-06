# Proposal: source-identity-memory-trust-promotion-acceptance-ledger-20260606

## Short Thesis

Add a read-only ledger for operator decisions on memory-trust promotion contract
records, keeping actual downstream priority promotion behind a future executor.

## What I Understood

The project is building a long-term memory loop for source identity trust. The
current contract can prove a promotion record is ready for operator review, but
the operator decision itself should be captured as evidence before any executor
or priority mutation exists.

## Proposed Solution

- Add `source_identity_calibration_memory_trust_promotion_acceptance_ledger.py`.
- Validate the promotion-contract envelope, status, safety flags, summary counts,
  contract records, and false-authority fields.
- Validate operator `accepted` / `rejected` records against contract records.
- Wire ledger status and counts into `main_pipeline_preview`.
- Register the ledger as `shadow_only` memory-learning evidence.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_trust_promotion_acceptance_ledger.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_memory_trust_promotion_acceptance_ledger.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-memory-trust-promotion-acceptance-ledger-20260606.md`

## Alternatives Considered

- Let contract-ready records immediately update downstream trust. Rejected
  because the executor, rollback apply path, and priority semantics are still
  separate.
- Store operator decisions only in external notes. Rejected because the Product
  Layer needs a typed evidence payload.
- Make promotion acceptance queueable. Rejected because it is memory-learning
  evidence, not a mixer action.

## Risks

- The ledger could be mistaken for actual promotion authority. Mitigate by
  forcing all execution, queue, model, runtime-memory, downstream-priority, and
  live authority flags false.
- Forged acceptance records could claim authority. Mitigate by rejecting unsafe
  flags and mismatched contract fields.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_acceptance_ledger.py tests/test_source_identity_calibration_memory_trust_promotion_contract.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_acceptance_ledger.py backend/source_identity_calibration_memory_trust_promotion_contract.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue this should become the executor. I would keep it as an
acceptance ledger because changing downstream priority affects proposal behavior
and still needs a distinct executor plus rollback apply proof.
