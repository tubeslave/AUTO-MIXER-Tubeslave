# Brief: WING MVP Current Evidence Latest Match Reports

## Task

Keep the current first WING MVP blocker evidence aligned with the latest
validated handoff and prestart summary match reports.

## Context

The active blocker remains `WING read-only connection proof` with status
`tooling_ready_waits_real_wing`. Project Director derives
`selected_evidence_files` from that blocker row, so stale evidence can steer the
next autonomous cycle toward older proof-boundary reports even when newer
validated reports exist.

## Scope

- Add the latest handoff/prestart match reports to the first blocker evidence.
- Assert the blocker table and Project Director expose those reports.
- Regenerate the latest MVP progress report.
- Record the decision and validation.

## Non-goals

- No backend or WING session contact.
- No live write path changes.
- No readiness closure.
- No change to the current first blocker status.

## Test plan

- Focused WING rails and Project Director tests.
- Project Director and direction guard JSON checks.
- Memory lint, diff whitespace check, no-WING quick check, and full pytest.
