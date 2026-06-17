# Brief: WING MVP Read-Only Runbook no-WING Manifest Sidecar Checklist

Date: 2026-06-15

## Task

Close the procedural gap where the WING read-only preflight runbook requires a
passed no-WING prestart `summary.json` but does not require the operator to
check the detached `artifact-manifest.json` summary-surface sidecar verdict
before opening the backend/WING WebSocket.

## Constraints

- Keep the first open blocker as `WING read-only connection proof`.
- Do not contact WING or widen live authority.
- Preserve `readiness_closure_allowed: false` and `live_authority_granted: false`
  semantics for offline no-WING evidence.
- Keep the patch limited to procedure, rails tests, and Council evidence docs.

## Expected Evidence

- Runbook checklist markers for the manifest summary-surface sidecar verdict.
- Rails test coverage that prevents the checklist from disappearing.
- Updated docs index, blocker evidence, generated latest progress, and report.
