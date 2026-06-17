# Brief: WING MVP Project Director Value Gate Self-Check Current Command Sync

## Task

Synchronize the Project Director selected action with the accepted-candidate
expected artifact trace value-gate summary-surface self-check requirements.

## Context

The no-WING summary now exposes and self-checks the value-gate summary surface,
but the standalone `## Current First Command` block did not include those
markers. Project Director selects that block as `selected_action`, so it could
continue directing operators with stale instructions even while the blocker
table and progress report mentioned the new self-check.

## Constraints

- No WING connection, OSC transport, backend WebSocket session, or live writes.
- Keep Project Director audit/planning-only.
- Do not close readiness from offline no-WING artifacts.
- Preserve the first open blocker: `WING read-only connection proof`.

## Done When

- `## Current First Command` includes the value-gate self-check markers.
- Project Director selected action includes those markers and reports no missing
  baseline markers.
- The current-first baseline marker tuple requires those fields.
- Focused and full validation pass.
