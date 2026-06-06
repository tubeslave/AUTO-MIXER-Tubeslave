# ADR: source-identity-calibration-acceptance-ledger-20260606

## Context

Automatic source typing is required before downstream source-specific mixing
decisions can be trusted. Replay calibration now produces preview-only operator
review candidates, but the system still needs a validated record of whether the
operator accepted or rejected those candidates. That record must not directly
write classifier models, long-term memory, downstream proposal priority, queues,
or live console state.

## Options Considered

1. Keep source identity calibration acceptance outside the repository evidence
   model.
2. Let accepted calibration records immediately update memory or downstream
   priority.
3. Add a separate read-only source identity calibration acceptance ledger.

## Decision

Use option 3. Add a transport-free acceptance ledger that validates operator
accept/reject records against source identity replay-calibration candidates and
surfaces the ledger in the main pipeline preview.

## Why This Won

A separate ledger keeps the lifecycle clear: replay calibration creates review
candidates, the operator records accept/reject decisions, and a later dedicated
persistence contract may consume accepted records. The ledger can be tested now
without granting model, memory, priority, queue, or live authority.

## Rejected Alternatives

- External-only acceptance was rejected because future memory and priority work
  would lack auditable repo evidence.
- Direct memory or downstream-priority writes were rejected because acceptance is
  only one proof gate and persistence needs a separate write contract.
- Queue import was rejected because calibration acceptance is not a supervised
  fader/gain apply action.

## Implementation Plan

1. Add a backend ledger module with strict candidate and record validation.
2. Reject unsafe authority claims, missing candidates, mismatched source types,
   mismatched trust states, unknown decisions, missing reviewers, and non-UTC
   timestamps.
3. Wire the ledger into `main_pipeline_preview` using optional server/config
   acceptance records and current replay-calibration candidates.
4. Register the ledger as `shadow_only` source identity evidence.
5. Update project memory and focused tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_acceptance_ledger.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_acceptance_ledger.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: accepted records could be misread as already-persisted source identity
trust. Mitigation: ledger safety and summary flags force `queue_mutation=false`,
`model_mutation=false`, `memory_mutation=false`,
`downstream_priority_mutation=false`, and `live_write=false`.

Rollback: remove the ledger call from `processing_pipeline_preview.py`, remove
the registry/capability rows, and keep replay calibration proposals as
preview-only candidates.
