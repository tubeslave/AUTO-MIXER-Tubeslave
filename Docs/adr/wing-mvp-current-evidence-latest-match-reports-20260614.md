# ADR: WING MVP Current Evidence Latest Match Reports

## Context

The first WING MVP blocker still waits for real WING read-only connection proof.
Recent validated reports made the handoff gate match verdicts, no-WING prestart
summary contract, and Current First Command prestart summary match flags
explicit. Project Director selects evidence from the blocker table, so the
blocker row needs to include these latest proof-boundary reports.

## Options considered

1. Leave the evidence list unchanged.
2. Add the latest validated match reports to the first blocker evidence list.
3. Make Project Director infer latest reports from `Docs/INDEX.md`.

## Decision

Use option 2.

## Why this won

It keeps the Project Director evidence source explicit, audited, and tied to the
current first blocker without changing runtime behavior or adding hidden report
selection logic.

## Rejected alternatives

- Option 1 leaves Goal-mode cycles pointed at an older evidence set.
- Option 3 makes report selection implicit and spreads blocker ownership into
  Project Director.

## Implementation plan

- Add the three 2026-06-14 match reports to the first blocker evidence cell.
- Assert their presence in blocker-table rail tests.
- Assert Project Director exposes them in `selected_evidence_files`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
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

Rollback removes the three report references and related tests/docs. This
change does not contact WING, mutate mixer state, grant live authority, or close
the readiness blocker.
