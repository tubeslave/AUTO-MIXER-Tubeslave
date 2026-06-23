# ADR: WING MVP no-WING Artifact Manifest Summary Surface

## 1. Context

The no-WING `artifact-manifest.json` is the detached hash authority for
`summary.json` and operator-readable `summary.md`. The main summaries should
also tell operators what manifest contract to expect.

## 2. Options considered

- Keep manifest details only in `artifact-manifest.json`.
- Expose the manifest contract in `summary.json` and `summary.md`.
- Add a sidecar hash to `summary.json`.

## 3. Decision

Expose required artifact IDs, paths, required SHA field names, and summary
surface status in `summary.json` and `summary.md`.

## 4. Why this won

It makes the no-WING handoff easier to inspect from the primary summary while
preserving the detached manifest as the final hash authority.

## 5. Rejected alternatives

Manifest-only review was rejected because it hides important contract fields from
the primary operator surface. Hashing the sidecar from `summary.json` was
rejected because it creates a recursive artifact relationship.

## 6. Implementation plan

- Add shared no-WING manifest contract constants.
- Render the required manifest surface in JSON and markdown summaries.
- Add tests and Project Director baseline markers.

## 7. Test plan

Run focused no-WING summary tests, Project Director/WING rail tests, full pytest,
`git diff --check`, and memory lint.

## 8. Risks and rollback

Risk is limited to stale wording or marker drift. Roll back by removing the
summary surface fields and matching baseline markers. This ADR does not contact
WING, grant live authority, or authorize fader/gain movement.
