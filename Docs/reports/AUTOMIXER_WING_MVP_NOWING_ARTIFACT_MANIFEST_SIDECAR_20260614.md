# WING MVP no-WING Artifact Manifest Sidecar

## Summary

Each no-WING release-check run now writes an `artifact-manifest.json` sidecar
that hashes the generated `summary.json` and operator-readable `summary.md`
artifacts.

## Changed

- `tools/no_wing_release_check.py` writes `artifact-manifest.json` after final
  summary artifacts are written.
- The manifest records path, byte count, and SHA256 for `summary.json` and
  `summary.md`.
- The manifest explicitly keeps `artifact_manifest_readiness_closure_allowed:
  false` and `artifact_manifest_live_authority_granted: false`.
- Project Director baseline coverage requires manifest path/count/hash and
  no-authority markers before supervised proof planning.

## Safety Boundary

This is an offline no-WING integrity artifact. It does not contact WING, query a
console, mutate mixer state, grant live authority, or authorize fader/gain
movement.

## Remaining Blocker

The WING MVP still requires a real WING read-only preflight capture and an
operator-reviewed accepted candidate before any supervised fader/gain proof run.
