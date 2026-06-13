# ADR: WING MVP Live Authority Disallowed By Packet Next Action

## Context

The current WING MVP blocker is still `WING read-only connection proof`. The
offline handoff packet is not a live execution token. It already emits
`live_authority_granted: false`, but the selected blocker action and supporting
docs did not require that field before supervised proof planning.

## Options considered

1. Leave the field only in the packet.
2. Promote `live_authority_granted: false` into the blocker action, runbook,
   MVP doc, autoloop docs, and tests.
3. Combine this with broader full-autonomous readiness changes.

## Decision

Use option 2.

## Why this won

The selected blocker and autoloop instructions are the operator-visible rail for
the next physical WING step. Requiring `live_authority_granted: false` there
prevents an offline handoff packet from being treated as authority to control
the console.

## Rejected alternatives

- Leaving the field only in generated packet output is too easy to miss in the
  next cycle.
- Combining with broader full-autonomous readiness changes would widen the
  diff without being required for this boundary.

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
