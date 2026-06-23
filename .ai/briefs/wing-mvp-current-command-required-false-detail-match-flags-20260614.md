# Brief: WING MVP Current Command Required-False Detail Match Flags

## Task

Require the Current First Command and synchronized WING MVP rails to demand
packet-vs-plan match verdicts for backend-port required-false detail fields.

## Context

The active WING MVP blocker remains `WING read-only connection proof`. The
no-WING handoff validator now compares backend-port no-authority detail fields
against the offline read-only plan, but Goal-mode routing still needs those
match verdicts in the selected action baseline.

## Scope

- Add the five required-false detail `*_matches_plan: true` markers to Current
  First Command baseline coverage.
- Promote the same markers into the blocker row, MVP docs, runbook, and
  autoloop prompt/readme.
- Add focused tests for Project Director, rails, prompt/readme, and progress
  visibility.
- Regenerate latest MVP progress.

## Non-goals

- Do not contact WING.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not grant live authority.
- Do not close the first WING blocker from offline evidence.
