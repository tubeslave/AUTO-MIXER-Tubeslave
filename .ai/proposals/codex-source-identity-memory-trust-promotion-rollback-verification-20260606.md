# Proposal: source-identity-memory-trust-promotion-rollback-verification-20260606

## Short Thesis

Add an operator-reviewed rollback verification gate so applied source identity
trust promotion cannot become downstream-priority evidence until rollback
readiness is explicitly verified.

## What I Understood

The apply audit records rollback targets, but it intentionally leaves
`rollback_verified=false`. The next step toward safe long-term learning is a
separate verifier that proves rollback evidence has been reviewed without
granting priority, queue, model, runtime memory, or live mixer authority.

## Proposed Solution

- Add `source_identity_calibration_memory_trust_promotion_rollback_verification.py`.
- Validate apply-audit type, method, status, timestamps, summary counts, audit
  records, and safety flags.
- Accept explicit operator verification records only when they match apply-audit
  records and carry reviewed rollback evidence.
- Emit verified records with `rollback_verified=true` and
  `usable_as_downstream_priority_gate_evidence=true`.
- Keep `downstream_priority_change_allowed_by_verification=false`; a separate
  downstream-priority gate must consume the evidence later.
- Wire Product Layer preview and register the method as non-live `shadow_only`
  memory-learning infrastructure.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_trust_promotion_rollback_verification.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_memory_trust_promotion_rollback_verification.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-memory-trust-promotion-rollback-verification-20260606.md`

## Alternatives Considered

- Mark rollback verified inside the apply audit. Rejected because audit and
  operator verification should be independently reviewable.
- Let verified rollback evidence directly change proposal priority. Rejected
  because that would skip the separate downstream-priority gate.
- Persist verification records immediately. Rejected because persistent trust
  store ownership and retention policy still need their own review.

## Risks

- Verification records may be mistaken for live authority. Mitigate with
  `can_apply_live=false`, `can_import_to_queue=false`, `live_write=false`, and
  no mixer imports.
- Forged records may try to smuggle priority or runtime-memory authority.
  Mitigate by validating record safety flags and failing closed.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_rollback_verification.py tests/test_source_identity_calibration_memory_trust_promotion_apply_audit.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_rollback_verification.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi might push for immediate downstream-priority consumption after rollback
verification. I would keep that as the next separate gate because priority
policy needs its own precedence, rollback, and operator visibility contract.
