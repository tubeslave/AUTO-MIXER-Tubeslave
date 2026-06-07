# ADR: WING Read-Only Preflight Session Confirmation Gate - 2026-06-07

## Context

The first supervised WING MVP blocker is a real read-only connection proof. The
preflight runner is allowed to query read-only state only after the operator has
prepared the intended backend/WING session. It must not become a tool that
silently runs against a stale, wrong, or unprepared session.

The existing operator channel confirmation verifies the inspected channel set.
It does not confirm that the backend/WING session itself is the intended
read-only preflight session.

## Options considered

1. Keep session readiness as runbook prose only.
2. Add an interactive confirmation prompt.
3. Add an explicit `--operator-confirm-session-ready` CLI flag required with
   `--require-physical-readback`.

## Decision

Use option 3. Physical readback preflight now requires
`--operator-confirm-session-ready` before any WebSocket connection opens. The
offline plan and real `summary.json` record:

- `operator_session_confirmation_required`
- `operator_confirmed_session_ready`

Evidence intake, generated handoff packets, and the no-WING release gate reject
artifacts or command templates that omit the confirmation.

## Why this won

The WING preflight workflow is artifact-driven. A CLI flag is visible in the
runbook, handoff packets, plan previews, release-check summaries, and captured
summary files. It is also testable without touching hardware.

## Rejected alternatives

Runbook-only confirmation is too easy to lose when a command is copied out of
context. Interactive prompting can be useful for ad hoc scripts, but it makes
the generated plan, handoff packet, and release-gate contract less explicit.

## Implementation plan

1. Add `--operator-confirm-session-ready` to the preflight runner.
2. Require it when physical readback is requested.
3. Include the confirmation in offline plans and real summaries.
4. Add `operator_session_confirmation_missing` to fail-closed plan conditions.
5. Reject missing confirmation in evidence intake, handoff packets, and no-WING
   release checks.
6. Update runbook, tests, ADR, report, and documentation index.

## Test plan

- Runner validation rejects physical preflight without session confirmation.
- Plan-only output records the required and confirmed session fields.
- Evidence validation rejects missing session confirmation fields.
- No-WING release checks reject stale capture, plan, and handoff commands.
- Focused pytest and quick no-WING release checks pass before commit.

## Risks and rollback

Risk: previously generated physical preflight commands without the new flag no
longer validate. This is acceptable because the first WING read-only blocker is
still open.

Rollback is a small revert of the runner flag, evidence validation, no-WING
checks, handoff generator, tests, and documentation. No mixer state is touched
by the change.
