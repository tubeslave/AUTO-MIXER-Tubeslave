# Brief: WING Read-Only Preflight Evidence Mixed Artifacts Fail Closed

Date: 2026-06-06

## Context

The supervised WING Rack MVP now has a read-only preflight runner and a
transport-free evidence evaluator for `wing_readonly_preflight.v1` summaries.
The evaluator may see multiple `summary.json` files under the artifact root.

## Task

Harden the evidence evaluator so a valid physical read-only preflight artifact
does not hide an invalid, corrupt, stale, or unsafe artifact in the same
evidence set.

## Safety Boundary

- No live OSC/WING writes.
- No WebSocket, OSC, or hardware connection.
- No readiness closure from this change.
- Evidence may expose only a manual-review candidate after the artifact set is
  clean.

## Success Criteria

- Mixed valid+invalid preflight artifacts produce a replacement-required status.
- `readiness_matrix_update_allowed` is false when any invalid artifact is
  present.
- Release-check validation expects a clean evidence set before a readiness
  update candidate is exposed.
- Focused tests and memory lint pass.
