# Brief: WING MVP Current Evidence Index Sync

## Task

Keep the first WING MVP blocker evidence synchronized with current WING MVP
reports indexed in `Docs/INDEX.md`.

## Context

The active blocker remains `WING read-only connection proof`. Project Director
uses that blocker evidence as `selected_evidence_files`, but several current
2026-06-14 WING reports were indexed without being available to the selected
evidence trail.

## Scope

- Add the missing current WING MVP 2026-06-14 report paths to the first blocker
  evidence.
- Add rail tests that compare `Docs/INDEX.md` current WING MVP 2026-06-14
  reports against the first blocker evidence and Project Director selected
  evidence.
- Keep the index-sync evidence trail current after the backend-port local socket
  helper report is added to the active WING blocker evidence.
- Record the decision and validation.

## Non-goals

- No WING/backend contact.
- No live write changes.
- No readiness closure.
- No change to the first blocker status.

## Test plan

- Focused WING rails and Project Director tests.
- Project Director and direction guard JSON checks.
- Memory lint, diff whitespace check, no-WING quick check, and full pytest.
