# ADR: WING MVP Doc Structured Gate

## Context

The WING read-only runbook, blocker rows, handoff packet, and no-WING release
summaries now use a structured `supervised_prestart_gate` contract. The
canonical MVP document still described the supervised proof boundary only in
terms of accepted-candidate prestart trace and proof-plan blockers.

## Options considered

1. Leave `Docs/MVP_WING_SUPERVISED.md` unchanged.
2. Add structured gate requirements to Stop-The-Line and Current Next Action.
3. Remove the older accepted-candidate trace wording and use only structured
   gate wording.

## Decision

Use option 2. The MVP document now stops supervised proof unless
`supervised_prestart_gate` exactly matches `expected_supervised_prestart_gate`
and does not grant live authority.

## Why this won

It keeps high-level policy aligned with the exact evidence now emitted by
no-WING release summaries while preserving the existing proof-plan blocker and
accepted-candidate trace requirements.

## Rejected alternatives

Leaving the document unchanged was rejected because it kept a weaker stop rule.
Removing the older wording was rejected because those gates still matter.

## Implementation plan

- Patch `Docs/MVP_WING_SUPERVISED.md`.
- Extend `tests/test_wing_mvp_rails.py`.
- Record validation in a report.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k mvp_doc`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py`
- Affected WING/no-WING tests.
- `python3 tools/memory/memory_lint.py`
- Full pytest suite.

## Risks and rollback

The main risk is doc wording drift if the structured gate field names change.
Rollback by updating the MVP doc and rails test in the same commit that renames
the handoff summary fields.
