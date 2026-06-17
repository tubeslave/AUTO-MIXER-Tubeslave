# WING Read-Only Runner Markdown Verifier Readiness Matrix Denial

## Context

The detached no-WING artifact manifest verifier fails if the operator-facing
Markdown summary claims
`artifact_manifest_verifier_readiness_matrix_update_allowed: true`, even when
the Markdown SHA is refreshed. The WING read-only preflight runner must mirror
that denial marker before opening any WebSocket connection.

## Options considered

- Keep requiring only readiness closure and live authority denial markers for
  the Markdown verifier surface.
- Also require the readiness-matrix update denial marker for the Markdown
  verifier surface.

## Decision

Require
`summary_md_artifact_manifest_verifier_readiness_matrix_update_allowed_denies_authority`
in the runner artifact-manifest verifier checks and in the WING MVP current
command.

## Why this won

Readiness matrix updates are a separate authority surface from readiness
closure and live authority. Mirroring the Markdown denial marker keeps the
operator-facing summary fail-closed before backend contact.

## Rejected alternatives

Relying on the verifier's general status was rejected because stale or partial
runner contracts should not be able to skip a specific readiness-matrix denial
marker.

## Implementation plan

- Add the marker to `ARTIFACT_MANIFEST_VERIFIER_REQUIRED_CHECKS`.
- Add it to the current WING blocker command and runner mirror list.
- Extend runner tests so a false marker blocks the run before WebSocket.

## Test plan

- Targeted runner contract and mismatch tests.
- No-WING quick release check.
- Detached artifact manifest verifier check on the generated no-WING artifact.
- Focused WING/no-WING safety suite.
- Full pytest before commit.

## Risks and rollback

Older no-WING artifacts without this marker will be rejected by the runner. That
is acceptable for the current WING proof rail. Roll back by removing this marker
from the runner, blocker command, tests, and this ADR only if a newer explicit
readiness-matrix denial contract replaces it.
