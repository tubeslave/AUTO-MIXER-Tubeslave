# WING MVP Progress Prestart Trace Gate

## Context

The proof-plan builder, processing registry, and no-WING release gate now require
accepted read-only preflight source ids to be backed by candidate review rows
with the backend-port prestart contract before supervised WING proof steps can
proceed. The MVP blocker table and generated progress report still surfaced the
next action as a generic no-WING quick check plus read-only preflight runbook.

## Task

Align the supervised WING MVP status rails so the first blocker explicitly
requires a valid accepted-candidate prestart trace before any supervised proof.

## Constraints

- Documentation/status tooling only.
- No backend startup.
- No WING transport usage.
- No OSC/WebSocket writes.
- Keep the first blocker status `tooling_ready_waits_real_wing`.

## Test Plan

- Focused MVP rails and Project Director tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- `git diff --check`.
- Full pytest suite before commit if time permits.
