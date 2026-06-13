# Brief: WING MVP Backend Prestart Template Failure Detail Fields

Date: 2026-06-13

## Task

Make the shared backend-port prestart failure-detail contract include the
generated backend-listener template status and authority-denial fields.

## Context

The no-WING busy-port summary already exposes pending template markers and
template authority denials. The shared
`backend_port_prestart_check_failure_detail_fields` contract still listed only
the base listener-review fields, so handoff and proof-plan validators did not
name the newer template fields as required failure evidence.

## Scope

- Extend `WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS`.
- Update WING MVP docs, runbook, autoloop docs, and generated latest progress.
- Add tests that assert the shared contract includes the template fields.

## Out Of Scope

- No WING hardware contact.
- No backend listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No readiness or blocker closure.
