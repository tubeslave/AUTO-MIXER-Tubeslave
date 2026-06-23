# ADR: no-WING Manifest Cross-Surface Subcheck Render

## Context

The no-WING release summary is an offline operator gate before any real WING
read-only session. Its manifest cross-surface checks protect the handoff from
stale or inconsistent manifest contract metadata.

The aggregate mismatch list was available, but the operator-readable Markdown
summary did not render the individual contract-id, digest, and digest-flag
subcheck verdicts.

## Options considered

1. Keep only the aggregate mismatch list.
2. Render individual cross-surface subcheck verdicts and require their Markdown
   lines.
3. Rewrite summary rendering with a schema-driven output layer.

## Decision

Choose option 2. The no-WING summary surface now requires and renders the
individual manifest cross-surface subcheck verdicts.

## Why this won

It improves operator diagnosis of offline gate drift while preserving the live
boundary: no console contact, no mixer mutation, no readiness closure, and no
live authority.

## Rejected alternatives

Option 1 hides useful diagnostic context. Option 3 is a broader refactor than
needed for this focused gate.

## Implementation plan

- Add four cross-surface subcheck fields to the required summary surface.
- Render the fields in `summary.md`.
- Replace their Markdown values after cross-surface validation computes actual
  verdicts.
- Update focused tests and fixtures.

## Test plan

- Targeted manifest summary/cross-surface tests.
- no-WING quick check.
- Project Director guard.
- Standard pytest.

## Risks and rollback

Risk is limited to offline report rendering. Roll back by removing the four
required fields, Markdown rows, post-check replacements, and focused test
updates.
