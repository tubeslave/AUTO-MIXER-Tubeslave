# Brief: WING MVP no-WING artifact manifest summary surface

## Task

Expose the no-WING artifact manifest contract in `summary.json` and the
operator-readable `summary.md`, not only in the detached `artifact-manifest.json`.

## Context

The sidecar manifest now declares required artifact IDs, paths, count matching,
and SHA availability for `summary.json` and `summary.md`. Operators and Project
Director should see the required manifest contract from the main no-WING summary
without needing to inspect the sidecar first.

## Constraints

- Do not contact WING or backend services.
- Do not grant live authority or readiness closure.
- Do not hash `artifact-manifest.json` from `summary.json`, because that would
  create a recursive artifact hash dependency.
- Keep the manifest contract offline and deterministic.

## Acceptance

- `summary.json` exposes the required manifest artifact IDs, paths, and SHA
  field names.
- `summary.md` renders the same operator-facing manifest surface.
- Project Director baseline requires the manifest summary surface markers.
- The sidecar manifest continues to hash final `summary.json` and `summary.md`.
