# ADR: WING MVP Current Evidence Listener Probe Error Report

## Context

The supervised WING Rack MVP is still blocked on real read-only connection
proof. The backend-port prestart listener-probe error field report documents
that owner-probe failures remain visible to the operator before real WING
preflight, but that report was not selected through the first blocker evidence
cell.

## Options considered

1. Leave the blocker evidence list unchanged.
2. Add the listener-probe error field report to the first blocker evidence list.
3. Make Project Director infer latest diagnostic reports from `Docs/INDEX.md`.

## Decision

Use option 2.

## Why this won

Project Director already treats the first blocker evidence cell as the selected
evidence source. Adding the report there keeps the active proof context explicit
without changing runtime behavior or adding report-selection inference.

## Rejected alternatives

- Option 1 keeps Project Director blind to the latest owner-probe diagnostic
  report.
- Option 3 adds hidden selection rules and could drift from the operator-owned
  blocker table.

## Implementation plan

- Add the listener-probe error field report to the first blocker evidence cell.
- Assert the report remains visible through WING MVP blocker parsing.
- Assert Project Director exposes the report in `selected_evidence_files`.
- Regenerate the latest MVP progress report and index the decision/report.

## Test plan

- Focused WING rails, Project Director, and Direction Guard tests.
- Project Director JSON check.
- Direction Guard JSON check.
- `git diff --check`.
- `tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

Rollback removes the report reference and related tests/docs. This does not
change live WING behavior, readiness status, or operator authority.
