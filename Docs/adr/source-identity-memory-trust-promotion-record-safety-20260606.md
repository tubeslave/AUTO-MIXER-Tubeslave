# ADR: source-identity-memory-trust-promotion-record-safety-20260606

## Context

The source identity memory trust promotion contract now validates promotion-plan
candidates, fresh replay context, and rollback evidence. Its parent object has a
safe `safety` envelope, but individual contract records are likely to be used by
operator-facing read models.

## Options Considered

1. Keep safety flags only on the parent contract.
2. Add record-level safety flags while preserving the parent envelope.
3. Convert contract records into executable promotion actions.

## Decision

Use option 2. Each contract record carries explicit read-only/no-live/no-query
flags while remaining non-executable.

## Why This Won

Record-level flags make partial rendering safer. A UI table, report row, or
future read model can show a record without implying live authority, queue
mutation, model mutation, runtime-memory mutation, or downstream-priority
mutation.

## Rejected Alternatives

- Parent-only safety was rejected because it depends on every consumer preserving
  the full envelope.
- Executable promotion actions were rejected because this cycle only hardens
  evidence shape. A separate executor and rollback apply path would be required.

## Implementation Plan

1. Add `read_only`, `live_write`, `mutates_console_state`, `queries_console`,
   `rollback_evidence_required`, and `live_widening_allowed` to contract
   records.
2. Add focused assertions in the contract tests.
3. Add Product Layer assertions so preview payloads preserve the fields.
4. Record the lesson in `.ai/memory/project.md`.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_contract.py tests/test_operator_product_state.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_contract.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: the extra fields are repetitive. That is acceptable because these records
are read-model evidence. Rollback is to remove the record-level fields and
their focused assertions, leaving the parent contract safety envelope intact.
