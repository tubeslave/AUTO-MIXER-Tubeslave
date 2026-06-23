# ADR: WING MVP no-WING Artifact Manifest SHA Availability

## 1. Context

The no-WING sidecar manifest hashes `summary.json` and `summary.md`. The next
handoff hardening step is to make hash availability and required artifact count
matching explicit instead of inferred.

## 2. Options considered

- Keep only raw SHA256 fields.
- Add explicit availability/count-match booleans to the manifest.
- Create a separate manifest schema file.

## 3. Decision

Add explicit SHA availability and required-count match fields directly to
`artifact-manifest.json`.

## 4. Why this won

The sidecar remains self-contained, deterministic, and easy to audit offline.
The fields also give Project Director stable marker strings for Current First
Command coverage.

## 5. Rejected alternatives

Raw SHA-only evidence was rejected because missing or stale fields would be
harder to diagnose in automation. A separate schema file was rejected as too
large for the current two-artifact no-WING bundle.

## 6. Implementation plan

- Add required-count and count-match fields in the manifest writer.
- Add SHA availability flags for `summary.json`, `summary.md`, and the required
  artifact set.
- Add tests and Project Director marker coverage.

## 7. Test plan

Run focused no-WING summary tests, focused Project Director/WING rail tests, full
pytest, `git diff --check`, and memory lint.

## 8. Risks and rollback

Risk is limited to stale marker wording. Roll back by removing the new manifest
fields and matching Project Director markers. This ADR does not contact WING,
grant live authority, or authorize fader/gain movement.
