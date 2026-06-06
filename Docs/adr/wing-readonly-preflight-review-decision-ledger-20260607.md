# ADR: WING Read-Only Preflight Review Decision Ledger

## Context

The first WING MVP blocker requires a real read-only preflight artifact and
manual review before readiness rows can be updated. The review report now shows
a checklist, but the operator decision also needs a durable, machine-readable
record so later readiness work can audit exactly which artifact was accepted or
rejected.

## Options considered

1. Keep the operator decision as prose in the review report.
2. Add a file-backed read-only decision ledger linked to clean preflight
   candidates.
3. Let a clean review report automatically close readiness rows.

## Decision

Add `backend/wing_readonly_preflight_review_decision.py`. The ledger reads
operator decision JSON files from
`artifacts/wing_readonly_preflight_review_decisions`, validates them against
clean WING preflight candidates, and reports accepted/rejected records without
granting live authority or closing readiness rows.

## Why this won

The decision becomes auditable and replayable while preserving the manual gate.
It also follows the existing acceptance-ledger pattern for physical readback
artifacts.

## Rejected alternatives

- Prose-only decision: not reliably machine-checkable.
- Automatic readiness closure: unsafe because operator acceptance is only one
  input to a separate readiness-matrix update.

## Implementation plan

1. Add a read-only decision ledger module.
2. Support direct decision records and JSON decision files.
3. Validate artifact source, run id, summary path, events path, UTC timestamp,
   checklist ids, and unsafe authority flags.
4. Add no-WING release-check validation and summary output.
5. Update focused tests and runbook documentation.

## Test plan

- `python3 -m py_compile backend/wing_readonly_preflight_review_decision.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk is limited to offline validation and reporting. Rollback is to remove the
ledger module, no-WING validation hook, tests, and documentation. No live WING
transport or write path is affected.
