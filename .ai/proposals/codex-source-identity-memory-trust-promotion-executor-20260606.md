# Proposal: source-identity-memory-trust-promotion-executor-20260606

## Short Thesis

Add an operator-gated, non-live source identity trust promotion executor that
can apply dry-run specs only to an explicit trust store, while Product Layer
preview remains non-mutating.

## What I Understood

The dry-run executor makes future source identity trust promotion concrete, but
it still does not apply anything. The project needs a narrow executor boundary
to prove the apply semantics before any persistent runtime store or downstream
priority policy uses those results.

## Proposed Solution

- Add `source_identity_calibration_memory_trust_promotion_executor.py`.
- Validate dry-run envelope, status, record fields, target paths, idempotency
  keys, duplicate IDs, and no-authority flags.
- Require `apply=true`, `operator_ack=true`, and an explicit mutable trust store
  before any trust state update.
- Wire Product Layer preview with `apply=false`.
- Register the executor as non-live `shadow_only` memory-learning
  infrastructure.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_trust_promotion_executor.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_memory_trust_promotion_executor.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-memory-trust-promotion-executor-20260606.md`

## Alternatives Considered

- Persist trust state directly in runtime configuration. Rejected because the
  persistent store and rollback audit are not reviewed yet.
- Keep dry-run only. Rejected because it does not prove apply semantics.
- Use the Product Layer task queue. Rejected because memory trust promotion is
  not a live mixer action and should not use mixer task execution.

## Risks

- The executor could be mistaken for live console authority. Mitigate by keeping
  `can_write_live=false`, no mixer imports, no queue mutation, and Product Layer
  preview `apply=false`.
- Applied trust may later influence proposals. Mitigate by requiring a separate
  reviewed persistent trust-store and rollback audit before downstream priority
  policy consumes it automatically.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_executor.py tests/test_source_identity_calibration_memory_trust_promotion_executor_dry_run.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_executor.py backend/source_identity_calibration_memory_trust_promotion_executor_dry_run.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi might argue this should immediately persist into runtime memory. I would
keep persistence separate because the rollback audit and downstream-priority
policy still need their own review gates.
