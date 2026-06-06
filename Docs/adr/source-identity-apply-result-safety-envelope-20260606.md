# ADR: source-identity-apply-result-safety-envelope-20260606

## Context

Source identity memory-trust promotion apply results can become rollback audit
evidence. The executor and audit summary deny live writes, console queries, and
downstream-priority authority, but future consumers may inspect individual
apply-result rows.

## Options Considered

1. Keep only executor-level safety fields.
2. Add a new downstream-priority policy gate.
3. Add row-level negative authority fields and validate them in the apply audit.

## Decision

Use option 3. Apply-result rows now explicitly deny console queries,
downstream-priority changes, and live widening, and the apply audit rejects
forged rows that claim those authorities.

## Why This Won

This keeps the existing architecture intact while making the rollback evidence
contract harder to misuse. It is small, fail-closed, and does not add any new
runtime authority.

## Rejected Alternatives

- Executor-level-only safety was rejected because row-level evidence can drift
  from summary-level safety.
- A new downstream-priority gate was rejected because priority policy promotion
  is a separate larger task.

## Implementation Plan

1. Add the three false safety fields to executor apply-result rows.
2. Require those fields in the apply-audit row validator.
3. Add focused regression coverage for forged row-level authority.
4. Record the durable lesson in project memory.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_apply_audit.py tests/test_source_identity_calibration_memory_trust_promotion_executor.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_apply_audit.py backend/source_identity_calibration_memory_trust_promotion_executor.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: hand-built test fixtures or external evidence rows that omit the new
fields will now fail validation. This is intended because missing safety
evidence is not proof.

Rollback: remove the three row-level fields from executor output, remove them
from the apply-audit validator, and drop the focused regression test.
