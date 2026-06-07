# Brief: WING Read-Only Preflight Session Confirmation Gate - 2026-06-07

## Context

The current supervised WING MVP blocker is still the real read-only connection
proof. The runner can already produce an offline plan and, when allowed, capture
physical readback through an existing backend/WING session.

## Problem

Physical readback preflight requires operator-approved channels, but the command
did not separately confirm that the intended backend/WING session is ready for
read-only inspection. A copied command could therefore be run against the wrong
or unprepared session while still carrying channel confirmation.

## Scope

- Require an explicit session-ready confirmation for physical read-only
  preflight.
- Record the confirmation in offline plans and real preflight summaries.
- Reject evidence and no-WING release artifacts that omit the confirmation.
- Update the operator runbook and local documentation.

## Out Of Scope

- Running WING hardware from Codex.
- Connecting or disconnecting the mixer.
- Granting live-write authority.
- Closing the WING read-only MVP blocker without operator-reviewed physical
  evidence.
