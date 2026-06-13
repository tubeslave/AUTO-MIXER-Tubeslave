# ADR: WING MVP Next Manual Artifact Required By Packet Next Action

## Context

The project is still on the first WING MVP blocker: `WING read-only connection
proof`. The handoff packet is an offline operator packet and must point to, not
replace, the next real WING proof artifact.

## Options considered

1. Keep the existing packet Markdown unchanged.
2. Render and require `next_manual_artifact_required: real
   wing_readonly_preflight.v1 summary.json` across current rails.
3. Run a real WING preflight now.

## Decision

Use option 2.

## Why this won

It makes the next required evidence explicit in both machine-readable JSON and
operator-facing Markdown while preserving the no-WING boundary.

## Rejected alternatives

- Option 1 leaves the key hidden from operator-facing handoff review.
- Option 3 is not allowed in this turn because no WING/backend session is being
  started or inspected.

## Implementation plan

- Render the explicit field in the handoff packet Markdown.
- Include it in no-WING release Markdown summaries.
- Update the WING MVP blocker action, MVP doc, read-only runbook, autoloop
  README, and cycle prompt.
- Add regression assertions and regenerate the MVP progress report.

## Test plan

- Focused packet, no-WING release, Project Director, and WING MVP rail tests.
- Project Director JSON check.
- Direction guard JSON check.
- `git diff --check`.
- `python3 tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

Rollback removes the explicit field rendering and related docs/tests. It does
not change live WING behavior.
