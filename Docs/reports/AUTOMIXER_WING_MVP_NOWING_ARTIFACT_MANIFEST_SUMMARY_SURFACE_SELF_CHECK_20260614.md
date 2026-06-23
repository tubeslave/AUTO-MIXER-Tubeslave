# WING MVP no-WING Artifact Manifest Summary Surface Self-Check

## Summary

The no-WING release summary now verifies that operator-readable `summary.md`
renders required artifact manifest contract lines.

## Changed

- `tools/no_wing_release_check.py` adds a fail-closed manifest summary surface
  self-check.
- `tests/test_no_wing_release_check.py` verifies both the passing surface and a
  missing-line failure.
- Project Director rails require the new rendered/missing self-check markers
  before supervised proof planning.

## Safety Boundary

This is an offline no-WING integrity contract. It does not contact WING, query a
console, mutate mixer state, grant live authority, or authorize fader/gain
movement.

## Remaining Blocker

The WING MVP still requires a real WING read-only preflight capture and an
operator-reviewed accepted candidate before any supervised fader/gain proof run.
