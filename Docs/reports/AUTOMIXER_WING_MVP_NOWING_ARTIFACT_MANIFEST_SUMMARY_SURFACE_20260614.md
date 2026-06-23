# WING MVP no-WING Artifact Manifest Summary Surface

## Summary

The no-WING `summary.json` and operator-readable `summary.md` now expose the
artifact manifest contract before an operator opens the detached
`artifact-manifest.json`.

## Changed

- `tools/no_wing_release_check.py` adds shared manifest contract constants and
  renders required manifest fields in JSON and markdown summaries.
- `tests/test_no_wing_release_check.py` verifies the new summary surface.
- Project Director rails require the manifest summary surface markers before
  supervised proof planning.

## Safety Boundary

This is an offline no-WING integrity contract. It does not contact WING, query a
console, mutate mixer state, grant live authority, or authorize fader/gain
movement.

## Remaining Blocker

The WING MVP still requires a real WING read-only preflight capture and an
operator-reviewed accepted candidate before any supervised fader/gain proof run.
