# WING Read-Only Manifest Verifier Readiness Matrix Denial

## Context

The WING read-only preflight runner now denies readiness-matrix updates before a real read-only artifact exists. The no-WING artifact manifest verifier is the detached pre-WebSocket gate that proves the no-hardware bundle is current, self-consistent, and non-authoritative before the backend/WING session opens.

## Options considered

1. Keep readiness-matrix denial only on the runner and handoff packet surfaces.
2. Extend the no-WING artifact manifest verifier summary, manifest, markdown, and verifier report contract with an explicit readiness-matrix denial.

## Decision

Add `artifact_manifest_verifier_readiness_matrix_update_allowed: false` to the no-WING verifier summary/manifest surfaces and `readiness_matrix_update_allowed: false` to the verifier report required fields.

## Why this won

The first WING blocker must not be closed or advanced by offline evidence. Putting the denial in the detached verifier prevents a stale or edited no-WING bundle from looking like a readiness-matrix update candidate before any WebSocket opens.

## Rejected alternatives

Relying on `readiness_closure_allowed: false` alone was rejected because the readiness matrix is a separate tracking surface with its own update semantics.

## Implementation plan

Update the no-WING summary/manifest constants, verifier expected surface, markdown denial checks, and tests that cover successful bundles and edited markdown authority claims.

## Test plan

Run targeted no-WING artifact verifier tests, no-WING quick validation, focused WING/no-WING tests, `git diff --check`, and the full standard pytest command.

## Risks and rollback

Risk is low because the new fields are false-valued reporting gates. Roll back by removing the field from the verifier constants, emitted surfaces, and tests if an older consumer rejects the additional key.
