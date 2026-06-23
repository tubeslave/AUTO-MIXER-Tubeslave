# ADR: no-WING Manifest Summary JSON Self-Check

## Context

The no-WING release summary is used as an offline operator gate before any WING
read-only session. It must not overstate completeness or readiness.

The manifest summary surface already reports required JSON fields and required
Markdown lines. Markdown line completeness was checked, but JSON field
completeness was reported as an empty missing list without a computed check.

## Options considered

1. Keep `artifact_manifest_summary_surface_missing_fields` as a declared empty
   list.
2. Compute missing JSON fields from the required field list before rendering.
3. Replace the summary surface with a larger schema framework.

## Decision

Choose option 2. The no-WING summary now recomputes missing required JSON
fields and fails the manifest summary surface check if any are absent.

## Why this won

It closes the stale self-report risk while keeping the WING boundary unchanged:
offline only, no console query, no mixer mutation, no readiness closure, and no
live authority.

## Rejected alternatives

Option 1 leaves the self-check too weak. Option 3 is more structural than this
specific gate needs.

## Implementation plan

- Add `apply_no_wing_artifact_manifest_summary_json_surface_check`.
- Reuse the existing manifest summary surface failed-check marker.
- Render the computed `artifact_manifest_summary_surface_missing_fields`.
- Add focused tests for missing JSON fields.

## Test plan

- Focused tests for summary embedding, missing Markdown line, missing JSON
  field, and missing digest line.
- no-WING quick check.
- Project Director check.
- Standard pytest.

## Risks and rollback

Risk is limited to offline reporting. Roll back by removing the JSON surface
helper and restoring the previous constant empty missing-field rendering.
