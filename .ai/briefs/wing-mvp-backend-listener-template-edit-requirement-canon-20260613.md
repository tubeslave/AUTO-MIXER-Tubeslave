# Brief: WING MVP Backend Listener Template Edit Requirement Canon

Date: 2026-06-13

## Task

Carry the reviewed backend-listener template requirement into the canonical
WING MVP blocker rail so Project Director and generated progress reports cannot
plan from the older pending-template-only contract.

## Context

The no-WING validator and read-only preflight runbook now require a
template-derived accepted confirmation to replace operator placeholders, set
`template_status: operator_reviewed_and_edited`, and set
`accepted_by_template: true`. The first WING MVP blocker still needs to expose
that requirement as the next-action contract.

## Scope

- Update `.ai/project-layer/wing_mvp_blockers.md`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Update Project Director/autoloop documentation and rail tests.
- Add an ADR and operator-facing report.

## Out Of Scope

- No WING hardware contact.
- No backend listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No blocker closure or readiness update.
