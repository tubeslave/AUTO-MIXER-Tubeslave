# Brief: WING MVP Backend Listener Template Authority Summary Fields

Date: 2026-06-13

## Task

Make the no-WING busy-port summary expose generated-template authority markers
that prove the backend-listener confirmation template cannot execute a backend
session or close readiness.

## Context

The no-WING gate generates `backend-listener-confirmation-template.json` when
an existing backend-port listener blocks preflight. The template already stores
`execution_allowed_by_template: false` and
`readiness_closure_allowed_by_template: false`, but the failed summary should
surface those markers directly for operator review and Project Director rails.

## Scope

- Add summary detail fields for generated-template execution authority.
- Add summary detail fields for generated-template readiness closure authority.
- Render those fields in `summary.md`.
- Update tests, runbook, blocker canon, ADR, and report evidence.

## Out Of Scope

- No WING hardware contact.
- No backend listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No readiness or blocker closure.
