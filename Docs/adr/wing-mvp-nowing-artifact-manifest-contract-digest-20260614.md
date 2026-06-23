# ADR: WING MVP no-WING Artifact Manifest Contract Digest

## 1. Context

The no-WING package now writes `artifact-manifest.json` and renders manifest
contract fields in `summary.json` and operator-readable `summary.md`. Those
surfaces need a stable proof that they describe the same contract without
creating a self-referential hash over `summary.json`.

## 2. Options considered

- Keep separate field assertions only.
- Hash generated summary artifacts into the summary.
- Add a canonical contract digest shared by all surfaces.

## 3. Decision

Add a canonical no-WING artifact manifest contract payload and SHA256 digest.
Render that digest in `summary.json`, `summary.md`, and `artifact-manifest.json`.

## 4. Why this won

The canonical payload is deterministic, does not depend on generated file bytes,
and still lets the final sidecar remain the authority for `summary.json` and
`summary.md` artifact hashes.

## 5. Rejected alternatives

Self-hashing the summary was rejected because it cannot be stable. A separate
validator sidecar was rejected as too much machinery for this handoff slice.

## 6. Implementation plan

- Define the canonical manifest contract payload and digest helper.
- Surface contract id, fields, digest, availability, and expected-match flag in
  all no-WING manifest review surfaces.
- Add fail-closed markdown coverage for missing digest lines.
- Add Project Director baseline markers.

## 7. Test plan

Run focused no-WING tests, WING rail tests, the full standard pytest command,
`git diff --check`, and memory lint.

## 8. Risks and rollback

Risk is limited to marker drift or digest payload churn. Roll back by removing
the digest fields and corresponding baseline markers. This ADR does not contact
WING, grant live authority, or authorize mixer writes.
