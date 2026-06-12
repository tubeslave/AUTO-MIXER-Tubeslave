# ADR: Project Goal Supervised AutoMix Structured Gate

## Context

The current WING proof path requires a structured handoff gate before supervised
physical proof can support widening. The high-level project goal capability
matrix still mentioned only accepted-candidate prestart trace for
`mode_supervised_automix`.

## Options considered

1. Leave the capability matrix unchanged.
2. Add the structured gate requirement to `mode_supervised_automix.next_step`.
3. Replace trace wording with only structured gate wording.

## Decision

Use option 2. `mode_supervised_automix.next_step` now requires
`supervised_prestart_gate` to match `expected_supervised_prestart_gate` in
addition to the accepted-candidate prestart trace.

## Why this won

It aligns the long-term project goal map with the active WING proof boundary
without changing runtime behavior or write authority.

## Rejected alternatives

Leaving the row unchanged was rejected because it understated the current proof
gate. Removing trace wording was rejected because trace evidence remains
required.

## Implementation plan

- Patch `backend/project_goal_capability_matrix.py`.
- Extend the matrix regression test.
- Record validation in a report.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_goal_capability_matrix.py -k supervised_automix`
- Affected project goal and WING tests.
- `python3 tools/memory/memory_lint.py`
- Full pytest suite.

## Risks and rollback

The main risk is goal-row wording drift if the structured gate names change.
Rollback by updating the matrix row and its regression in the same commit that
renames the handoff summary fields.
