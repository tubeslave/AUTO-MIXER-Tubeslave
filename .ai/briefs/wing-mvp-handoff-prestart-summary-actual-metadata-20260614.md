# Brief: WING MVP handoff prestart summary actual metadata

## Task

Make the WING read-only handoff packet reflect the actual metadata from a real
no-WING prestart `summary.json` when that file is provided.

## Context

The handoff packet carries `backend_port_prestart_summary_*` fields that later
gates compare against the no-WING prestart contract. Before this slice, the
packet used expected constants for those fields even when `--prestart-summary`
pointed at an existing file.

## Constraints

- Do not contact WING hardware.
- Do not open a backend/WING WebSocket.
- Do not grant execution, readiness, or live authority from failed prestart
  metadata.
- Preserve placeholder behavior when the prestart summary file is not present.

## Desired outcome

If an existing no-WING prestart summary says `status: failed` or carries
`environment_blocked_checks`, the handoff packet preserves that status and
detail instead of rendering the expected `passed` contract as if it were actual
evidence.
