# Brief: WING handoff prestart summary command binding

## Context

The WING read-only handoff packet already passes a no-WING prestart summary
placeholder into preview and capture commands. The no-WING summary should also
prove that each command's `--prestart-summary` value matches the validated plan
before any real WING session.

## Task

Make the handoff validator fail closed and report explicit preview/capture
prestart-summary command bindings.

## Acceptance

- The handoff validator reports `preview_command_prestart_summary_matches_plan`.
- The handoff validator reports `capture_command_prestart_summary_matches_plan`.
- Mismatched preview or capture `--prestart-summary` values fail no-WING
  validation.
- Project Director, WING rails, runbook, and MVP docs surface the new guard.
- No backend, WING, OSC, or live-write behavior is added.
