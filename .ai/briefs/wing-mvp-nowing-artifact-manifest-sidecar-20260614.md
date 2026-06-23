# Brief: WING MVP no-WING artifact manifest sidecar

## Task

Add a no-WING sidecar manifest that hashes the generated `summary.json` and
operator-readable `summary.md` artifacts.

## Context

The no-WING handoff now binds the proof-plan summary to `summary.md`, but the
bundle still benefits from a detached manifest that can hash both generated
files without recursive digest fields inside either artifact.

## Constraints

- Do not contact WING or backend services.
- Do not grant live authority or readiness closure.
- Keep manifest generation offline and after summary artifacts are written.
- Preserve Project Director focus on the first WING blocker.

## Acceptance

- `write_summary()` writes `artifact-manifest.json`.
- The manifest contains SHA256 values for `summary.json` and `summary.md`.
- The manifest declares no live authority and no readiness closure.
- Project Director baseline requires the manifest path, artifact count, summary
  hashes, and no-authority flags.
