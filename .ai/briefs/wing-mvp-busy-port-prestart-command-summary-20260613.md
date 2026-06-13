# Brief: WING MVP Busy-Port Prestart Command Summary

Date: 2026-06-13
Owner: Codex.app

## Task

Make the failed busy-port no-WING summary explicitly carry the shared
`backend_port_prestart_check_command` in both JSON and Markdown.

## Context

The WING read-only connection proof still waits for real hardware evidence. The
no-WING backend-port gate can fail closed before any intended backend/WING
session starts. When that happens, the operator should see the exact prestart
rerun command in the first failed summary, not only in downstream handoff or
proof-plan artifacts.

## Scope

- Add the shared prestart command to the `backend_port_free` failure detail.
- Render that command in the no-WING `summary.md`.
- Update WING MVP rails so the failed busy-port no-WING summary and handoff
  summary both name the same command contract.
- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend listeners.
- Do not grant live authority or close readiness.

## Validation

- Focused no-WING/WING MVP rail tests.
- no-WING quick gate with expected fail-closed busy-port output.
- Project Director, direction guard, memory lint, and full pytest before commit.
