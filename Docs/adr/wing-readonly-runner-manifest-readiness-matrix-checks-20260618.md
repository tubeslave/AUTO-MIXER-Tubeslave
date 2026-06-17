# WING Read-Only Runner Manifest Readiness Matrix Checks

## Context

The supervised WING MVP keeps readiness updates closed until a real read-only
WING artifact and manual review are accepted. The detached no-WING artifact
manifest verifier now proves that the manifest and human summaries deny
readiness matrix updates. The read-only preflight runner must mirror those
checks before opening any WebSocket connection.

## Options considered

- Trust the detached verifier status only.
- Require the runner to mirror the specific readiness-matrix denial checks in
  its prestart summary contract.

## Decision

Require `wing_readonly_preflight_runner.py` to include the verifier's
readiness-matrix denial checks in `ARTIFACT_MANIFEST_VERIFIER_REQUIRED_CHECKS`.

## Why this won

The runner is the final gate before backend contact. Mirroring the exact denial
checks keeps readiness matrix updates fail-closed even if a verifier report is
stale, incomplete, or missing a newly required safety marker.

## Rejected alternatives

Trusting only the detached verifier status was rejected because a `passed`
status without mirrored required checks would let the runner miss a contract
drift before WebSocket.

## Implementation plan

- Add the manifest, summary JSON, and summary Markdown readiness-matrix denial
  checks to the runner required-check tuple.
- Add tests that require those checks before WebSocket.
- Extend the verifier mismatch regression so a false readiness-matrix denial
  check blocks the run before WebSocket.

## Test plan

- Targeted runner tests for required verifier checks and pre-WebSocket blocking.
- No-WING quick release check.
- Focused WING/no-WING safety test group.
- Full pytest before merge in an environment that can complete the repository
  suite.

## Risks and rollback

The change can block a run if the detached verifier no longer emits the required
checks. That is intentional fail-closed behavior. Roll back by removing the
three required check names and this ADR if the verifier contract is deliberately
replaced by a newer safety mechanism.
