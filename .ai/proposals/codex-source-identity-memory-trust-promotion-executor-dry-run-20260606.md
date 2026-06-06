# Proposal: source-identity-memory-trust-promotion-executor-dry-run-20260606

## Short Thesis

Add a read-only executor dry-run that turns accepted source-identity memory
trust promotion execution candidates into deterministic apply specs while
keeping actual trust promotion behind a future operator-gated executor.

## What I Understood

The promotion execution plan identifies what could be executed later. The
system still needs an auditable, idempotent spec for that future operation:
target store, target path, previous and target trust state, rollback strategy,
and idempotency key.

## Proposed Solution

- Add `source_identity_calibration_memory_trust_promotion_executor_dry_run.py`.
- Validate the execution plan envelope, status, safety flags, summary counts,
  candidate fields, duplicate source specs, and false-authority fields.
- Convert valid execution candidates into dry-run apply specs for
  `source_identity_downstream_gate`.
- Wire dry-run status and counts into `main_pipeline_preview`.
- Register the dry run as `shadow_only` memory-learning evidence.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_trust_promotion_executor_dry_run.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_memory_trust_promotion_executor_dry_run.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-memory-trust-promotion-executor-dry-run-20260606.md`

## Alternatives Considered

- Apply trust promotion immediately. Rejected because executor safety,
  rollback application, and downstream priority semantics are not proven.
- Keep only the execution plan. Rejected because the future executor would lack
  a typed operation spec.
- Import dry-run specs into the Product Layer queue. Rejected because memory
  trust promotion is not a mixer action and this dry run is evidence only.

## Risks

- The dry run could be mistaken for real executor authority. Mitigate by
  forcing all apply, queue, model, runtime-memory, downstream-priority, and live
  authority flags false.
- Duplicate source specs could produce ambiguous trust updates. Mitigate by
  rejecting duplicate valid candidates for the same source candidate.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_executor_dry_run.py tests/test_source_identity_calibration_memory_trust_promotion_execution_plan.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_executor_dry_run.py backend/source_identity_calibration_memory_trust_promotion_execution_plan.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue the next step should apply to the downstream gate. I would keep
this as a dry-run because applying changes proposal behavior and needs a
separate executor contract plus rollback proof.
