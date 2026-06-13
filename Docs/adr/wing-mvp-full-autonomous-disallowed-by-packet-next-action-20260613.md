# ADR: WING MVP Full Autonomous Disallowed By Packet Next Action

## Context

The current WING MVP blocker is still `WING read-only connection proof`. The
long-term project objective includes full autonomous mixing, but the present
offline handoff packet is only operator guidance for a later read-only physical
preflight. It already emits `full_autonomous_ready: false`, but the selected
blocker action and supporting docs did not require that field before supervised
proof planning.

## Options considered

1. Leave the field only in the packet output.
2. Promote `full_autonomous_ready: false` into the blocker action, runbook,
   MVP doc, autoloop docs, and tests.
3. Add broader full-autonomous readiness work.

## Decision

Use option 2.

## Why this won

The selected blocker and autoloop instructions are the main rail for future
goal-mode cycles. Requiring `full_autonomous_ready: false` there keeps the
offline packet from being mistaken for proof that full autonomous WING control
is ready.

## Rejected alternatives

- Leaving the field only in generated packet output is too implicit for the
  next cycle.
- Broader full-autonomous readiness work would violate the Project Director's
  current first-blocker scope.

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
