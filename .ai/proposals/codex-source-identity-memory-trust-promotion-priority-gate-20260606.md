# Proposal: source-identity-memory-trust-promotion-priority-gate-20260606

## Short Thesis

Add a source identity memory-trust promotion priority gate so verified rollback
evidence can become explicit downstream-priority eligibility evidence without
changing proposal priority, queues, models, memory, trust stores, or live state.

## What I Understood

The rollback verifier proves that applied trust promotion has operator-reviewed
rollback evidence. The project still needs a separate boundary before that
evidence can influence proposal policy. This step should say "eligible for a
future policy consumer" and no more.

## Proposed Solution

- Add `source_identity_calibration_memory_trust_promotion_priority_gate.py`.
- Validate rollback-verification type, method, status, timestamp, summary
  counts, verified records, duplicate IDs, and safety flags.
- Reuse `evaluate_source_identity_downstream_trust` for final source-identity
  trust eligibility.
- Emit read-only priority-gate records with
  `downstream_priority_eligible=true` only for trusted downstream-gate decisions.
- Keep `downstream_priority_change_allowed_by_gate=false`, `can_import_to_queue=false`,
  `can_apply_live=false`, and every live/model/queue/runtime mutation flag false.
- Wire Product Layer preview and register the method as non-live `shadow_only`
  memory-learning infrastructure.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_trust_promotion_priority_gate.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_memory_trust_promotion_priority_gate.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-memory-trust-promotion-priority-gate-20260606.md`

## Alternatives Considered

- Let rollback verification directly feed priority policy. Rejected because
  rollback proof and policy eligibility are separate safety decisions.
- Mutate proposal priority inside the gate. Rejected because policy ordering
  needs its own consumer, precedence rules, audit trail, and rollback.
- Treat any verified record as eligible. Rejected because the shared source
  identity downstream gate must still block held, blocked, unknown, or stale
  evidence.

## Risks

- Eligible records may be mistaken for priority mutation authority. Mitigate
  with `policy_gate_only=true`, `direct_downstream_priority_mutation=false`,
  and a required separate proposal-priority consumer.
- Forged rollback-verification payloads may try to smuggle queue, model,
  runtime-memory, trust-store, or live authority. Mitigate by validating envelope,
  summary, and record-level safety flags before emitting records.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_priority_gate.py tests/test_source_identity_calibration_memory_trust_promotion_rollback_verification.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_priority_gate.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi may argue that a verified rollback record should immediately change
downstream proposal priority. I would keep the new gate as eligibility evidence
only because policy ordering is behavioral authority and needs its own reviewed
consumer.
