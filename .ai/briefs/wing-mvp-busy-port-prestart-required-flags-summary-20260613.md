# Brief: WING MVP Busy-Port Prestart Required Flags Summary

Date: 2026-06-13
Owner: Codex.app

## Task

Make the failed busy-port no-WING summary explicitly carry
`backend_port_prestart_check_required: true` and
`backend_port_prestart_check_must_pass_before_starting_backend: true` in both
JSON and Markdown.

## Context

The WING read-only connection proof still waits for real hardware evidence. The
no-WING backend-port gate can fail closed before any intended backend/WING
session starts. When that happens, the operator should see that the prestart
check is required and must pass before backend/WING startup in the first failed
summary, not only in downstream handoff artifacts.

## Scope

- Add the shared prestart required/must-pass flags to the `backend_port_free`
  failure detail.
- Render those flags in the no-WING `summary.md`.
- Update WING MVP rails so the failed busy-port no-WING summary and handoff
  summary both name the same required/must-pass contract.
- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend listeners.
- Do not grant live authority or close readiness.

## Validation

- Focused no-WING/WING MVP rail tests.
- no-WING quick gate with expected fail-closed busy-port output.
- Project Director, direction guard, memory lint, and full pytest before commit.
