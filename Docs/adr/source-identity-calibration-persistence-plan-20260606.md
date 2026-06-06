# ADR: source-identity-calibration-persistence-plan-20260606

## Context

Source identity replay calibration and its acceptance ledger now provide
operator-reviewed trust evidence. The system still needs a safe step between
accepted records and any future model, memory, or downstream-priority write. A
manual persistence plan creates that evidence boundary without granting write
authority.

## Options Considered

1. Let accepted calibration records update memory or downstream priority.
2. Keep accepted records in the ledger only.
3. Add a separate read-only source identity calibration persistence plan.

## Decision

Use option 3. Add a transport-free persistence plan that revalidates the
acceptance ledger and surfaces accepted calibration records as manual
persistence candidates only.

## Why This Won

A separate plan keeps the lifecycle explicit: replay calibration creates
candidates, the acceptance ledger records operator decisions, and the persistence
plan identifies which accepted records could feed a future write contract. The
plan can fail closed on forged or stale ledger data without writing memory,
models, downstream priority, queues, or live console state.

## Rejected Alternatives

- Direct memory or priority writes were rejected because acceptance alone is not
  a persistence contract.
- Ledger-only planning was rejected because future integrations need a second
  validation boundary.
- Waiting for a writer was rejected because a writer should consume audited
  candidates, not raw acceptance records.

## Implementation Plan

1. Add a backend persistence-plan module with strict ledger validation.
2. Validate ledger envelope, status, UTC timestamps, safety flags, summary
   counts, summary identity lists, summary safety claims, and accepted records.
3. Surface manual persistence candidates only when the ledger is clean and all
   candidates have been reviewed.
4. Wire the plan into `main_pipeline_preview`.
5. Register the plan as `shadow_only` source identity evidence.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_persistence_plan.py tests/test_source_identity_calibration_acceptance_ledger.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_persistence_plan.py backend/source_identity_calibration_acceptance_ledger.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: manual persistence candidates could be misread as already-written
calibration state. Mitigation: plan safety and summary flags force
`calibration_persistence_write_allowed_by_plan=false`,
`model_mutation_allowed_by_plan=false`, `memory_mutation_allowed_by_plan=false`,
`downstream_priority_change_allowed_by_plan=false`, and `live_write=false`.

Rollback: remove the plan call from `processing_pipeline_preview.py`, remove the
registry/capability rows, and keep acceptance-ledger records as read-only
evidence only.
