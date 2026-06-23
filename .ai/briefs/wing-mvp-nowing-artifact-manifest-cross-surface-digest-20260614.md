# Brief: WING MVP no-WING artifact manifest cross-surface digest

## Task

Make `artifact-manifest.json` prove that the manifest contract id and digest
rendered in no-WING `summary.json` and operator-readable `summary.md` match the
canonical manifest contract digest.

## Context

The no-WING bundle already writes a detached artifact manifest and exposes a
canonical manifest contract digest. The next hardening step is to make the final
sidecar state whether both review surfaces carry the same digest, so operators
and Project Director can detect surface drift before any supervised proof.

## Constraints

- Do not contact WING, backend services, or network targets.
- Do not grant readiness closure or live authority.
- Keep `artifact-manifest.json` as the final authority for artifact hashes.
- Preserve offline no-WING behavior and fail closed in documentation rails.

## Acceptance

- `summary.json` and `summary.md` declare the cross-surface digest contract
  requirement and no-authority boundary.
- `artifact-manifest.json` records per-surface match flags for contract id,
  contract SHA256, and expected-match booleans.
- The manifest records an empty mismatch list and true aggregate match for a
  valid run.
- Tests fail when the cross-surface match marker is absent or false in Current
  First Command baseline coverage.
