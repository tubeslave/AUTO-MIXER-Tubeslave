# Brief: WING MVP no-WING artifact manifest required set

## Task

Make the no-WING `artifact-manifest.json` declare the expected artifact IDs and
paths, not only an artifact count.

## Context

The manifest sidecar hashes `summary.json` and `summary.md`. To be reviewable as
a handoff contract, it should also expose the expected artifact set and whether
the generated set matches it.

## Constraints

- Do not contact WING or backend services.
- Do not grant live authority or readiness closure.
- Keep the manifest offline and deterministic.

## Acceptance

- Manifest records required and actual artifact IDs.
- Manifest records required and actual artifact paths.
- Manifest records present/match booleans for both sets.
- Project Director baseline requires those manifest set markers.
