# Proposal: source-identity-memory-trust-promotion-apply-audit-20260606

## Short Thesis

Add a read-only rollback audit for applied source identity memory-trust
promotion so applied trust-store changes are reviewable before any downstream
priority policy can consume them.

## What I Understood

The executor proves a bounded apply path, but applied trust is still too
powerful to feed downstream proposal priority directly. The next safe step is a
transport-free audit layer that records rollback evidence and rejects forged
authority.

## Proposed Solution

- Add `source_identity_calibration_memory_trust_promotion_apply_audit.py`.
- Validate executor type, method, status, timestamps, safety flags, summary
  counts, apply result fields, target paths, and duplicate IDs.
- Emit audit records only when the executor status is applied and valid apply
  results exist.
- Record rollback target state from the executor's previous trust state.
- Wire Product Layer preview to expose the audit but keep zero records because
  preview calls the executor with `apply=false`.
- Register the audit as non-live `shadow_only` memory-learning infrastructure.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_trust_promotion_apply_audit.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_memory_trust_promotion_apply_audit.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-memory-trust-promotion-apply-audit-20260606.md`

## Alternatives Considered

- Let the executor audit itself. Rejected because downstream consumers need a
  separate evidence artifact that can be validated independently.
- Persist rollback state immediately. Rejected because persistent trust storage
  and downstream-priority policy still need separate review gates.
- Treat applied trust as proposal-priority authority. Rejected because this
  would collapse the safety boundary before rollback evidence is reviewed.

## Risks

- Audit records could be mistaken for downstream-priority permission. Mitigate
  with explicit false flags for priority mutation, live authority, queue
  mutation, model mutation, and runtime-memory mutation.
- Forged executor payloads could smuggle live authority. Mitigate by validating
  executor and result safety flags and failing closed as replacement-required.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_apply_audit.py tests/test_source_identity_calibration_memory_trust_promotion_executor.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_apply_audit.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi might argue the audit should persist rollback records now. I would keep
persistence separate until the target trust store and downstream priority gate
have their own ownership and rollback contract.
