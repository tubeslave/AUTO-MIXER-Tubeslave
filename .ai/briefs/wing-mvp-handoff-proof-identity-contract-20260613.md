# WING MVP Handoff Proof Identity Contract

## Context

The WING read-only preflight expected plan already carries proof identity:
`proof_artifact_schema`, `proof_runner`, `selected_mvp_blocker`, and
`selected_mvp_blocker_status`. The handoff packet currently binds allowed
requests, response types, fail-closed conditions, and blocker anchor metadata,
but does not expose the same proof identity fields directly.

## Task

Make the WING read-only preflight handoff packet declare the proof artifact
schema, proof runner, selected MVP blocker, and selected blocker status. Extend
the no-WING release gate so the handoff packet must match both the shared
contract and the expected plan artifact.

## Constraints

- No backend startup.
- No WING transport usage.
- No OSC/WebSocket writes.
- No readiness closure from offline packet evidence.
- Preserve the current supervised WING MVP blocker boundary.

## Test Plan

- Focused handoff/no-WING pytest modules.
- no-WING quick gate with a local artifact root.
- Project direction and WING rail tests.
- `git diff --check`.
- `tools/memory/memory_lint.py`.
- Standard pytest suite.
