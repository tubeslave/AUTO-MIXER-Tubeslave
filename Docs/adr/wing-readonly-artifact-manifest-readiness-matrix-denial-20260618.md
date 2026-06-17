# WING Read-Only Artifact Manifest Readiness Matrix Denial

## Context

The no-WING verifier report now denies readiness-matrix updates before any WING WebSocket opens. The detached `artifact-manifest.json` contract still needed the same denial at the artifact-manifest level, matching the existing readiness-closure and live-authority denial fields.

## Options considered

1. Rely only on verifier report field `readiness_matrix_update_allowed: false`.
2. Add artifact-manifest and cross-surface readiness-matrix denial fields.

## Decision

Add `artifact_manifest_readiness_matrix_update_allowed: false` and `artifact_manifest_contract_cross_surface_readiness_matrix_update_allowed: false`, with JSON, Markdown, sidecar manifest, and verifier checks.

## Why this won

The readiness matrix is updated from real proof only. The detached manifest is the pre-WebSocket bundle integrity gate, so it must deny readiness-matrix updates independently of the runner and verifier report.

## Rejected alternatives

Keeping the denial only on runner or verifier report surfaces was rejected because stale or edited artifact manifests could still lack a direct readiness-matrix denial.

## Implementation plan

Extend no-WING artifact-manifest contract fields, cross-surface checks, emitted summary/manifest values, Markdown rendering, detached verifier checks, and operator-facing current-command markers.

## Test plan

Run targeted artifact-manifest contract tests, no-WING quick, focused WING/no-WING tests, `git diff --check`, and the full standard pytest command.

## Risks and rollback

Risk is low because the new fields are false-valued reporting gates. Roll back by removing the new fields and their assertions if a legacy consumer rejects the additional manifest keys.
