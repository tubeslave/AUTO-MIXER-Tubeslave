# ADR: WING MVP Current Evidence Latest Reports

## Context

The WING MVP first blocker still waits for real WING read-only proof. The latest
validated reports define two current boundaries: shared read-only artifact
identity and the stricter accepted-decision prestart ready gate.

## Options considered

1. Leave the first blocker evidence list unchanged.
2. Add the latest validated reports to the first blocker evidence list.
3. Derive current reports from `Docs/INDEX.md` inside Project Director.

## Decision

Use option 2.

## Why this won

The blocker table is already Project Director's source for
`selected_evidence_files`. Adding the latest reports there keeps the current
proof context explicit without changing runtime behavior or adding inference
logic.

## Rejected alternatives

- Option 1 lets future cycles miss the newest proof-boundary reports.
- Option 3 spreads ownership into Project Director and makes report selection
  implicit.

## Implementation plan

- Add the two 2026-06-13 WING MVP reports to the first blocker evidence cell.
- Assert their presence through blocker table parsing.
- Assert Project Director exposes them in `selected_evidence_files`.
- Regenerate the latest MVP progress report and index this change.

## Test plan

- Focused WING MVP rails, Project Director, and guard tests.
- Project Director JSON check.
- Direction guard JSON check.
- `git diff --check`.
- `python3 tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

Rollback removes the two report references and related tests/docs. It does not
change live WING behavior, readiness status, or operator authority.
