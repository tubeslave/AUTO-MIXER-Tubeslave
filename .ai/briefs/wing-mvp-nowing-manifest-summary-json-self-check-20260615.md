# Brief: no-WING Manifest Summary JSON Self-Check

## Task

Make the no-WING manifest summary surface compute missing JSON fields from the
required manifest contract instead of declaring an empty list by construction.

## Context

The no-WING `summary.md` self-check already fails closed when required
operator-readable manifest lines are missing. The paired `summary.json`
surface also exposes `artifact_manifest_summary_surface_missing_fields`, but
that list was not recomputed from the required JSON fields.

## Constraints

- Stay offline and no-WING.
- Do not contact or query a console.
- Do not mutate mixer state.
- Do not close WING readiness or grant live authority.

## Acceptance

- `artifact_manifest_summary_surface_missing_fields` is computed from
  `NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_SUMMARY_SURFACE_REQUIRED_FIELDS`.
- A missing required JSON field marks the no-WING artifact manifest summary
  surface check failed.
- Existing Markdown missing-line behavior remains unchanged.
