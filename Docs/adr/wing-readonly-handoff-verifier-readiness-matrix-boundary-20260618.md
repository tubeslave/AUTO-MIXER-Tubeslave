# WING Read-Only Handoff Verifier Readiness Matrix Boundary

## Context

The current WING first command requires no-WING verifier surfaces to deny
readiness matrix updates before any backend/WING session. The no-WING release
artifacts already expose `artifact_manifest_verifier_readiness_matrix_update_allowed:
false`, but the operator handoff packet only displayed readiness closure and
live authority denial for that verifier surface.

## Options considered

- Leave the handoff packet with only readiness closure and live authority
  fields.
- Display the readiness-matrix denial alongside the other verifier safety
  boundary fields.

## Decision

Add `artifact_manifest_verifier_readiness_matrix_update_allowed: false` to the
handoff packet JSON and rendered Markdown safety boundary.

## Why this won

The handoff packet is the operator-facing checklist before real WING work.
Readiness matrix updates are a separate authority surface, so the packet must
show that the detached verifier grants no readiness update authority.

## Rejected alternatives

Relying on the no-WING artifact alone was rejected because the operator handoff
would not visibly mirror the full no-authority boundary from the current WING
command.

## Implementation plan

- Add the field to `tools/build_wing_readonly_preflight_handoff_packet.py`.
- Render the field in handoff Markdown.
- Add JSON and Markdown assertions in
  `tests/test_wing_readonly_preflight_handoff_packet.py`.

## Test plan

- Targeted handoff packet tests.
- No-WING quick release check.
- Focused WING/no-WING safety suite.
- Full pytest before commit.

## Risks and rollback

The change is additive and read-only. Roll back by removing the field, Markdown
line, tests, and this ADR if a newer handoff contract replaces the verifier
readiness-matrix boundary.
