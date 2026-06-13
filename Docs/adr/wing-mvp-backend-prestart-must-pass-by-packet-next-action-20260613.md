# ADR: WING MVP Backend Prestart Must Pass By Packet Next Action

## Context

The current WING MVP blocker is still `WING read-only connection proof`. The
offline handoff packet already emits
`backend_port_prestart_check_must_pass_before_starting_backend: true`, while the
selected blocker action focuses on the detailed structured prestart gate.

## Options considered

1. Leave the field only in generated packet output.
2. Promote `backend_port_prestart_check_must_pass_before_starting_backend: true`
   into the blocker action, runbook, MVP doc, autoloop docs, and tests.
3. Execute backend listener checks now.

## Decision

Use option 2.

## Why this won

The top-level field makes the pass-before-start requirement operator-visible
without contacting hardware or widening live authority. It prevents a future
cycle from treating the backend prestart check as optional or post-start.

## Rejected alternatives

- Leaving the field only in generated output is too implicit for goal-mode
  cycles.
- Executing listener checks now could interfere with local backend/WING session
  state and is not required for this offline rail update.

## Implementation plan

- Update the WING MVP blocker action and current command.
- Add stop-rule and runbook requirements.
- Update Codex autoloop README and cycle prompt.
- Add regression assertions for Project Director and WING MVP rail surfaces.
- Regenerate the MVP progress report.

## Test plan

- Focused Project Director and WING MVP rail tests.
- Project Director JSON check.
- Direction guard JSON check.
- `git diff --check`.
- `python3 tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

Rollback removes the requirement from docs and tests. It does not change packet
generation or live behavior.
