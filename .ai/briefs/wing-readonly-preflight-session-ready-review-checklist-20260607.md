# Brief: WING Read-Only Preflight Session-Ready Review Checklist - 2026-06-07

## Context

The first supervised WING MVP blocker is still the real read-only connection
proof. The preflight runner now records operator confirmation that the intended
backend/WING session is ready for physical read-only inspection.

## Problem

The operator review checklist and accepted decision ledger did not require a
separate checklist item for the session-ready confirmation. A real artifact
could carry `operator_confirmed_session_ready: true`, but the human review
record could still omit that explicit review step.

## Scope

- Add `confirm_operator_session_ready` to the required review checklist.
- Surface session-ready confirmation in artifact review rows and markdown.
- Make accepted decision records without the new checklist id fail closed.
- Keep handoff packets and no-WING release checks aligned with the stricter
  review contract.

## Out Of Scope

- Running WING hardware.
- Closing the read-only proof blocker from offline evidence.
- Granting live authority or widening supervised write scope.
