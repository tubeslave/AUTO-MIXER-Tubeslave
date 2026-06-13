# Brief: WING MVP Backend Listener Template Pending Summary Fields

Date: 2026-06-13

## Task

Make the no-WING busy-port summary expose pending generated-template markers
that prove the backend-listener confirmation template is not yet accepted.

## Context

The no-WING gate generates `backend-listener-confirmation-template.json` when
an existing backend-port listener blocks preflight. The template already stores
`accepted_by_template: false` and placeholder operator fields, but the failed
summary should surface those markers directly for operator review.

## Scope

- Add summary detail fields for generated-template accepted-by-template status.
- Add summary detail fields for placeholder operator fields.
- Render those fields in `summary.md`.
- Update tests, runbook, ADR, and report evidence.

## Out Of Scope

- No WING hardware contact.
- No backend listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No readiness or blocker closure.
