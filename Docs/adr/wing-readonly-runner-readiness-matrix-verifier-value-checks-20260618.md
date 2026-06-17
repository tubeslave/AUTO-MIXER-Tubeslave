# WING Read-Only Runner Readiness Matrix Verifier Value Checks

## Context

The WING read-only preflight runner is the final gate before any WebSocket
connection. The detached no-WING artifact manifest verifier already reports
whether summary JSON, summary Markdown, manifest, and verifier report surfaces
all keep `readiness_matrix_update_allowed` false. The runner must mirror those
value checks before backend contact.

## Options considered

- Keep checking only readiness closure and live authority values.
- Also require the readiness-matrix update value checks emitted by the detached
  verifier.

## Decision

Add the detached verifier's `readiness_matrix_update_allowed_matches_expected`
checks to `ARTIFACT_MANIFEST_VERIFIER_REQUIRED_CHECKS` in
`tools/wing_readonly_preflight_runner.py`.

## Why this won

Readiness closure and readiness matrix updates are separate gates. A real WING
artifact can only move the readiness matrix after accepted manual review, so the
runner must fail closed if the verifier report does not prove that readiness
matrix updates are denied before WebSocket.

## Rejected alternatives

Relying on closure/live-authority checks alone was rejected because it would
leave a stale-contract path where readiness matrix update authority was not
mirrored by the runner prestart summary.

## Implementation plan

- Require the three surface value checks for `summary_json`, `summary_md`, and
  `manifest`.
- Require the verifier report value check.
- Extend runner tests to cover required checks and WebSocket-blocking mismatch
  behavior.

## Test plan

- Targeted runner contract and mismatch tests.
- No-WING quick release check.
- Detached artifact manifest verifier check on the generated no-WING artifact.
- Focused WING/no-WING safety test group.
- Full pytest before merge in an environment that can complete the repository
  suite.

## Risks and rollback

The runner may block on older no-WING artifacts that do not contain these
checks. That is acceptable for the active WING proof rail. Roll back by removing
the four required check names and this ADR only if the verifier contract is
replaced by a newer explicit readiness-matrix denial mechanism.
