# ADR: WING MVP no-WING Artifact Manifest Required Set

## 1. Context

The no-WING sidecar manifest hashes `summary.json` and `summary.md`. It should
also declare that those are the required artifacts so count-only validation does
not become ambiguous.

## 2. Options considered

- Keep only artifact count.
- Add required IDs and paths to the manifest.
- Create a separate manifest schema file.

## 3. Decision

Add required and actual artifact IDs/paths plus present booleans directly to
`artifact-manifest.json`.

## 4. Why this won

It makes the manifest self-describing without another file and keeps the handoff
bundle easy to review offline.

## 5. Rejected alternatives

Count-only was rejected because it cannot prove the specific artifact set. A
separate schema file was rejected as unnecessary for the current two-artifact
bundle.

## 6. Implementation plan

- Add required artifact ID/path lists in the manifest writer.
- Add present booleans for IDs and paths.
- Add tests and Project Director markers.

## 7. Test plan

Run focused no-WING summary tests, focused Project Director/WING rail tests, full
pytest, `git diff --check`, and memory lint.

## 8. Risks and rollback

Risk is limited to marker drift. Roll back by removing the required-set fields
and matching Project Director markers. This ADR does not grant live WING
authority.
