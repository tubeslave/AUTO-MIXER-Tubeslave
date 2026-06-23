# WING MVP no-WING Artifact Manifest Cross-Surface Digest

## Summary

`artifact-manifest.json` now checks that the no-WING manifest contract id and
SHA256 digest shown in `summary.json` and operator-readable `summary.md` match
the canonical manifest contract digest.

## Changed

- `tools/no_wing_release_check.py` adds a cross-surface helper for the manifest
  contract id, digest, and expected-match flags.
- `summary.json` and `summary.md` declare the required surfaces, required
  fields, expected match, and no-authority boundary.
- `artifact-manifest.json` records per-surface match flags, an aggregate match
  flag, and a mismatch list.
- Project Director baseline markers now require the cross-surface match before
  supervised proof planning.

## Safety Boundary

This is an offline no-WING integrity check. It does not contact WING, query a
console, mutate mixer state, grant live authority, or authorize fader/gain
movement.

## Remaining Blocker

The WING MVP still requires a real `wing_readonly_preflight.v1` capture and an
operator-reviewed accepted candidate before any supervised fader/gain proof run.
