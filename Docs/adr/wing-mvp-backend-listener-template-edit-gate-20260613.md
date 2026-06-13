# ADR: WING MVP Backend Listener Template Edit Gate

Date: 2026-06-13
Status: accepted

## Context

The no-WING backend-port gate can generate a pending listener confirmation
template. The template includes placeholder operator fields and
`template_status: pending_operator_edit`; those fields must not clear the gate
if the operator only edits the accepted status and confirmation boolean.

## Options considered

1. Accept any JSON with matching PID and command line.
2. Reject generated-template metadata entirely.
3. Accept template-derived JSON only after pending markers are replaced.

## Decision

Use option 3. Template-derived confirmation must replace placeholder review
fields, move `template_status` to `operator_reviewed_and_edited`, and set
`accepted_by_template: true`.

## Why this won

It preserves the template as an operator aid while keeping the acceptance
boundary explicit and review-backed.

## Rejected alternatives

Blind acceptance was too weak because placeholders are non-empty strings.
Rejecting all template metadata would remove useful provenance from the
operator-edited confirmation artifact.

## Implementation plan

Add fail-closed validation in `tools/no_wing_release_check.py`, cover partial
and full template edits in tests, and update runbook/report evidence.

## Test plan

Run focused no-WING and WING rail tests, memory lint, direction guard, and the
standard full pytest command.

## Risks and rollback

If this blocks legitimate operator-confirmed backend sessions, remove only the
template-origin metadata checks while keeping placeholder rejection and the
existing PID/command-line checks.
