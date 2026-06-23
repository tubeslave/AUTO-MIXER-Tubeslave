# Brief: WING MVP no-WING artifact manifest contract digest

## Task

Bind the no-WING artifact manifest contract with a stable canonical digest that
is visible in `summary.json`, operator-readable `summary.md`, and
`artifact-manifest.json`.

## Context

The no-WING bundle now writes a detached `artifact-manifest.json` and renders the
manifest contract surface in the operator summary. The next hardening step is to
make those surfaces prove they are using the same canonical manifest contract,
without hashing `summary.json` into itself.

## Constraints

- Do not contact WING or backend services.
- Do not grant readiness closure or live authority.
- Keep the digest independent of generated artifact file hashes.
- Preserve `artifact-manifest.json` as the final hash authority for
  `summary.json` and `summary.md`.

## Acceptance

- `summary.json` exposes the manifest contract id, field list, digest, and
  expected-digest match flag.
- `summary.md` renders those same contract digest fields.
- `artifact-manifest.json` exposes the same contract id, field list, digest, and
  expected-digest match flag.
- Missing digest lines fail the operator-readable manifest surface check.
- Project Director baseline requires the digest markers before supervised proof.
