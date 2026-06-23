# WING MVP no-WING Artifact Manifest SHA Availability

## Summary

The no-WING `artifact-manifest.json` now explicitly states that the required
two-artifact bundle has matching artifact count and available SHA256 values for
`summary.json` and operator-readable `summary.md`.

## Changed

- `tools/no_wing_release_check.py` adds required-count, count-match, and SHA
  availability fields to the sidecar manifest.
- `tests/test_no_wing_release_check.py` verifies the new sidecar fields against
  generated artifacts.
- Project Director rails require the new manifest SHA/count markers before
  supervised proof planning.

## Safety Boundary

This is an offline no-WING integrity contract. It does not contact WING, query a
console, mutate mixer state, grant live authority, or authorize fader/gain
movement.

## Remaining Blocker

The WING MVP still requires a real WING read-only preflight capture and an
operator-reviewed accepted candidate before any supervised fader/gain proof run.
