# ADR: WING MVP Current Evidence Index Sync

## Context

The first WING MVP blocker still waits for real WING read-only proof. Project
Director selects evidence from the blocker table, while `Docs/INDEX.md` lists
the current WING MVP reports that describe the active proof boundary. Several
indexed 2026-06-14 reports were not present in the blocker evidence trail.

## Options considered

1. Leave index and blocker evidence independent.
2. Add the missing current indexed reports and test the narrow index-sync
   contract.
3. Make Project Director merge evidence from `Docs/INDEX.md` at runtime.

## Decision

Use option 2.

## Why this won

The blocker row remains the explicit audited source for Project Director
evidence, while tests prevent it from drifting behind the current WING MVP
index entries.

## Rejected alternatives

- Option 1 lets Project Director miss indexed proof-context reports.
- Option 3 hides evidence selection inside runtime inference and makes the
  blocker table less authoritative.

## Implementation plan

- Add missing `AUTOMIXER_WING_MVP_*_20260614.md` index reports to the first
  blocker evidence list.
- Add tests that derive the expected current report names/paths from
  `Docs/INDEX.md`.
- Assert the first blocker evidence and Project Director selected evidence
  contain the full current indexed set.
- Record validation in the task report.

## Test plan

- Focused WING rails and Project Director tests.
- Project Director JSON check.
- Direction guard JSON check.
- `tools/memory/memory_lint.py`.
- `git diff --check`.
- no-WING quick check.
- Full pytest.

## Risks and rollback

Rollback removes the added report references and sync tests. This does not
contact WING, mutate console state, grant live authority, or close the current
blocker.
