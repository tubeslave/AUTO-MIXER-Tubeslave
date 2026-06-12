# ADR: WING MVP Index Structured Gate

## Context

The active WING MVP proof boundary now includes a structured handoff gate:
`supervised_prestart_gate` must match `expected_supervised_prestart_gate`. The
main docs index still pointed mainly to the older final prestart gate report.

## Options considered

1. Leave `Docs/INDEX.md` unchanged.
2. Add structured gate reports to the reports table only.
3. Add structured gate reports to the reports table and the Current WING MVP
   rails row.

## Decision

Use option 3. The structured runbook and no-WING summary reports are active
operator/reviewer surfaces and belong in the Current WING MVP rails row.

## Why this won

The index is the canonical documentation entrypoint. It should expose the same
machine-readable gate that current runbooks, release summaries, blocker rows,
and goal matrix require.

## Rejected alternatives

Leaving the index unchanged was rejected because it hid the current gate.
Reports-only linking was rejected because the structured gate is an active rail,
not only historical evidence.

## Implementation plan

- Patch `Docs/INDEX.md`.
- Extend `tests/test_wing_mvp_rails.py`.
- Record verification in a report.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k docs_index_points`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py`
- Affected WING/no-WING tests.
- `python3 tools/memory/memory_lint.py`
- Full pytest suite.

## Risks and rollback

The main risk is an overly long Current WING MVP rails row. Roll back by moving
the linked reports into a dedicated WING MVP index while preserving these links
from the canonical index.
