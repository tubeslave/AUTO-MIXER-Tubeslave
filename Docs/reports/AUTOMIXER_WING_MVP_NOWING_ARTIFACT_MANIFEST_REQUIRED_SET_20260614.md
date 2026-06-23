# WING MVP no-WING Artifact Manifest Required Set

## Summary

The no-WING `artifact-manifest.json` now declares the expected artifact IDs and
paths for `summary.json` and operator-readable `summary.md`, then records whether
the generated manifest contains that required set.

## Changed

- `tools/no_wing_release_check.py` adds required/actual artifact ID and path
  fields to the sidecar manifest.
- `tests/test_no_wing_release_check.py` verifies the required set and present
  booleans.
- Project Director rails require the manifest set markers before supervised
  proof planning.

## Safety Boundary

This is an offline no-WING integrity contract. It does not contact WING, query a
console, mutate mixer state, grant live authority, or authorize fader/gain
movement.

## Remaining Blocker

The WING MVP still requires a real WING read-only preflight capture and an
operator-reviewed accepted candidate before any supervised fader/gain proof run.
