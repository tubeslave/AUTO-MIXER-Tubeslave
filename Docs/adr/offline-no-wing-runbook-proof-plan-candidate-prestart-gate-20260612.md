# Offline no-WING Runbook Proof Plan Candidate Prestart Gate

## Context

The no-WING release gate now requires `next_wing_proof_plan` to expose accepted candidate prestart trace fields. If accepted read-only preflight source ids exist but the candidate prestart trace is invalid, the proof plan must include `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`.

## Options considered

1. Leave runbook success criteria unchanged.
2. Add the new candidate-prestart fields and blocker to the existing Next WING Proof Plan success criteria.
3. Create a separate WING proof-plan runbook.

## Decision

Use option 2. The offline no-WING validation runbook now states that physical WING proof steps are not acceptable unless every accepted review source id is backed by a candidate review row with the backend-port prestart contract.

## Why this won

It keeps the operator checklist aligned with the current release gate while preserving the no-WING, no-live-write boundary.

## Rejected alternatives

- Leaving the wording generic was rejected because it hides the concrete blocker.
- A separate runbook was unnecessary for this scoped success-criteria update.

## Implementation plan

- Update `Docs/OFFLINE_NO_WING_VALIDATION_RUNBOOK.md`.
- Add a test that asserts the runbook includes the candidate-prestart fields and blocker.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "offline_no_wing_runbook_lists_wing_proof_candidate_prestart_gate"`
- Affected no-WING tests.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

The only risk is overly specific wording in the runbook. Rollback is safe by removing the added checklist bullets and test. No runtime behavior or console authority is affected.
