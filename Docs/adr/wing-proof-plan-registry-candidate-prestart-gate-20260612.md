# WING Proof Plan Registry Candidate Prestart Gate

## Context

`supervised_wing_proof_plan` now blocks physical proof steps unless an accepted read-only preflight decision is bound to a candidate review row with the backend-port prestart contract. The processing registry still described the method as a generic read-only checklist.

## Options considered

1. Leave registry metadata unchanged.
2. Update the existing method metadata with the candidate-prestart blocker and note.
3. Add a new registry method for the candidate-prestart gate.

## Decision

Use option 2. The existing `supervised_wing_proof_plan` registry row now names the candidate-prestart gate and keeps the method shadow-only, transport-free, and non-executing.

## Why this won

It aligns the machine-readable inventory with the current safety boundary without introducing a new method id or runtime behavior.

## Rejected alternatives

- Unchanged metadata was rejected because it could mislead status/readiness views.
- A new method id was unnecessary for a gate inside the existing proof-plan method.

## Implementation plan

- Update `backend/processing_registry.py` method metadata and UI next-step wording.
- Extend `tests/test_processing_registry.py`.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_processing_registry.py -k "supervised_wing_proof_plan_is_shadow_only_not_executor"`
- Affected WING/no-WING tests.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk is limited to inventory wording. Rollback is safe by restoring the previous summary, blocker, note, and test assertions. No console authority changes.
