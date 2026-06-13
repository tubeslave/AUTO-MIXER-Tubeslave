# Brief: WING MVP Backend Prestart Template Review Fields

Date: 2026-06-13
Owner: Codex.app

## Task

Keep the first open MVP blocker, `WING read-only connection proof`, on its
fail-closed path by making the shared backend-port prestart failure-detail
contract include all operator-review fields already emitted by the no-WING
busy-port summary.

## Context

The no-WING gate can already show generated listener-template review details:
safe-to-stop denial, live-authority denial, operator placeholder fields, and
required template edits. The shared
`backend_port_prestart_check_failure_detail_fields` list only named part of that
template evidence, so handoff packets, runbooks, Project Director, and tests did
not have one complete list to compare.

## Scope

- Extend `WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS`.
- Update rail docs and Project Director source text to use the complete field list.
- Add a validated report and ADR.
- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend listeners.
- Do not grant live authority or close readiness.

## Validation

- Focused WING/no-WING tests.
- no-WING quick gate, expected fail-closed when a local backend port is busy.
- Project Director and direction guard.
- Full pytest before commit.
