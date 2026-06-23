# WING MVP no-WING Artifact Manifest Contract Digest

## Summary

The no-WING artifact manifest contract now has a canonical SHA256 digest shared
by `summary.json`, operator-readable `summary.md`, and `artifact-manifest.json`.

## Changed

- `tools/no_wing_release_check.py` defines a stable no-WING manifest contract
  payload and digest.
- `summary.json`, `summary.md`, and `artifact-manifest.json` expose the contract
  id, field list, digest, availability flag, and expected-match flag.
- The operator-readable manifest surface self-check now fails closed when the
  contract digest line is missing.
- Project Director rails require the digest markers before supervised proof
  planning.

## Safety Boundary

This is an offline no-WING package integrity check. It does not contact WING,
query a console, mutate mixer state, grant live authority, or authorize
fader/gain movement.

## Remaining Blocker

The WING MVP still requires a real `wing_readonly_preflight.v1` capture and an
operator-reviewed accepted candidate before any supervised fader/gain proof run.
