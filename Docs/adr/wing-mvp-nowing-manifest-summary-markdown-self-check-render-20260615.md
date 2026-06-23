# ADR: no-WING Manifest Summary Markdown Self-Check Render

## Context

The no-WING release summary is an offline operator gate used before any real
WING read-only session. It must not imply readiness or surface completeness
unless the generated artifacts actually prove those claims.

The manifest summary surface check already computed missing Markdown field
lines, but the final Markdown output still rendered the affected self-check
rows with static success values.

## Options considered

1. Keep the static Markdown rows.
2. Replace the affected Markdown rows with computed values after the surface
   check runs.
3. Rewrite the whole no-WING Markdown summary writer.

## Decision

Choose option 2. The writer now updates the manifest summary Markdown
self-check rows after validation computes missing line evidence.

## Why this won

It removes a stale operator-facing success signal without changing the live
boundary: no console contact, no mixer mutation, no readiness closure, and no
live authority.

## Rejected alternatives

Option 1 leaves the operator-readable artifact weaker than the JSON summary.
Option 3 is a larger refactor that is not required to close this gate.

## Implementation plan

- Add a small helper to replace a named Markdown summary field line.
- Use it after `apply_no_wing_artifact_manifest_summary_surface_check`.
- Add a regression test that forces a missing required Markdown line and
  verifies `summary.md` renders the computed failure details.

## Test plan

- Focused manifest summary surface tests.
- no-WING quick check.
- Project Director guard.
- Standard pytest.

## Risks and rollback

Risk is limited to offline report rendering. Roll back by removing the helper,
the post-check line replacements, and the focused regression test.
