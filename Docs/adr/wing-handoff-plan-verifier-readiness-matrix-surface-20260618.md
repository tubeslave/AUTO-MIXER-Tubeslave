# WING Handoff Plan Verifier Readiness Matrix Surface

## Context

The no-WING bundle already exposes artifact-manifest verifier no-authority
fields, including `artifact_manifest_verifier_readiness_matrix_update_allowed:
false`. The narrower
`wing_readonly_handoff_plan_verifier_summary_surface_*` contract still mirrored
only readiness closure and live authority denial.

## Options considered

- Keep readiness-matrix denial only on the artifact-manifest verifier surface.
- Add readiness-matrix denial to the handoff-plan verifier summary surface too.

## Decision

Add
`wing_readonly_handoff_plan_verifier_summary_surface_readiness_matrix_update_allowed:
false` to the no-WING summary, sidecar manifest, detached verifier report, and
preflight-runner verifier gate.

## Why this won

The handoff-plan verifier surface is inspected before any backend/WING
WebSocket opens. Readiness-matrix update authority is separate from readiness
closure and live authority, so the same denial must be explicit on this surface.

## Rejected alternatives

Relying on the broader artifact-manifest verifier surface was rejected because
summary consumers could inspect the handoff-plan verifier surface independently
and miss the readiness-matrix denial.

## Implementation plan

- Extend `WING_READONLY_HANDOFF_PLAN_VERIFIER_SUMMARY_SURFACE_FIELDS`.
- Render the new field in no-WING `summary.md`.
- Require the field in `tools/verify_no_wing_artifact_manifest.py`.
- Require the corresponding verifier check in `tools/wing_readonly_preflight_runner.py`.
- Add focused regression assertions.

## Test plan

- Focused no-WING release-check tests.
- Focused WING preflight-runner verifier tests.
- no-WING quick check and detached manifest verifier.
- Full pytest before commit.

## Risks and rollback

This is additive and fail-closed. Roll back by removing the field from the
surface list, renderer, verifier expectations, runner checks, tests, and this
ADR if a later contract consolidates verifier no-authority fields elsewhere.
