# ADR: WING MVP no-WING Artifact Manifest Cross-Surface Digest

## 1. Context

The no-WING artifact bundle exposes a canonical manifest contract digest in
`summary.json`, operator-readable `summary.md`, and `artifact-manifest.json`.
Those surfaces need a sidecar-level check proving that the displayed contract id
and digest agree across the generated package.

## 2. Options considered

- Keep independent digest fields only.
- Add a full manifest schema validator.
- Add a targeted cross-surface digest match check in `artifact-manifest.json`.

## 3. Decision

Add a targeted cross-surface digest match check. The summary surfaces declare
the requirement and no-authority boundary, while the final sidecar records the
actual match flags and mismatch list.

## 4. Why this won

The check is deterministic, small, and aligned with the existing no-WING
artifact sidecar. It verifies the operator-readable and machine-readable
surfaces without introducing a self-referential hash.

## 5. Rejected alternatives

Independent fields were rejected because they do not expose a single sidecar
verdict. A full schema validator was deferred because the immediate need is the
canonical contract digest link, not a broader schema framework.

## 6. Implementation plan

- Add cross-surface constants and helper logic in the no-WING release checker.
- Render cross-surface requirement fields in `summary.json` and `summary.md`.
- Record per-surface match flags, mismatch list, aggregate match, and
  no-authority booleans in `artifact-manifest.json`.
- Add Project Director baseline markers and rail tests.

## 7. Test plan

Run focused no-WING tests, Project Director/WING rail tests, the full standard
pytest command, the no-WING runtime check, `git diff --check`, and memory lint.

## 8. Risks and rollback

Risk is limited to marker drift or an over-strict string check. Roll back by
removing the cross-surface fields, helper, and baseline markers. This ADR does
not contact WING, grant live authority, or authorize mixer writes.
