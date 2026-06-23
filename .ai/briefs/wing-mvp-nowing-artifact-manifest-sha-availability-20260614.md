# Brief: WING MVP no-WING artifact manifest SHA availability

## Task

Make the no-WING `artifact-manifest.json` explicitly declare that required
artifact hashes are available and that the generated artifact count matches the
required set.

## Context

The manifest sidecar already records SHA256 values for `summary.json` and
operator-readable `summary.md`. Reviewers and Project Director need explicit
boolean markers so a bundle cannot look complete when hashes or the required
artifact count are missing.

## Constraints

- Do not contact WING or backend services.
- Do not grant live authority or readiness closure.
- Keep the manifest offline and deterministic.
- Preserve the existing no-WING artifact set: `summary.json` and `summary.md`.

## Acceptance

- Manifest records required artifact count.
- Manifest records whether generated artifact count matches required count.
- Manifest records SHA availability for both required artifacts.
- Project Director baseline requires the new manifest SHA/count markers.
