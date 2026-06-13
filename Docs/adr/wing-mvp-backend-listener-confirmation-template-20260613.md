# ADR: WING MVP Backend Listener Confirmation Template

Date: 2026-06-13
Status: accepted

## Context

The no-WING prestart gate fails closed when `127.0.0.1:8765` already has a
listener. Operators can clear that condition only by confirming the listener is
the intended Automixer backend, or by clearing an unexpected listener after
review. The accepted confirmation JSON must match the current PID and command
line.

## Options considered

1. Keep only the runbook example JSON.
2. Generate a pending template from the observed listener diagnostics.
3. Generate an accepted confirmation automatically.

## Decision

Use option 2. Busy-port fail-closed summaries write a pending
`backend-listener-confirmation-template.json` seeded with the current listener
diagnostics.

## Why this won

The template reduces transcription risk while preserving the operator boundary.
It is not accepted by default and cannot clear the gate until the operator edits
the status, confirmation flag, review timestamp, and reviewer name.

## Rejected alternatives

Runbook-only JSON leaves the operator to copy PID and command-line values
manually. Automatic acceptance would grant too much authority from a diagnostic
tool and was rejected.

## Implementation plan

Add template builders to `tools/no_wing_release_check.py`, attach the template
path/status to busy-port summaries, update runbook/MVP rails, and cover the
flow with focused tests.

## Test plan

Run focused no-WING, Project Director, direction guard, and MVP rail tests, then
the standard backend pytest command before commit.

## Risks and rollback

If the template causes operator confusion, remove the template writer and keep
the existing accepted-confirmation validator. Do not relax the fail-closed
backend-port check or readiness rules.
