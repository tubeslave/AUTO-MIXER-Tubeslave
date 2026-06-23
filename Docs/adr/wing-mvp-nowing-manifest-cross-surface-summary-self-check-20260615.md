# ADR: no-WING Manifest Cross-Surface Summary Self-Check

## Context

The no-WING release summary is an offline operator gate before any real WING
read-only session. Its `summary.json`, `summary.md`, and detached
`artifact-manifest.json` must agree on the manifest contract digest before a
human treats the package as a coherent handoff.

The detached manifest already checked written files, but the primary summary
surfaces initialized the cross-surface verdict fields as successful.

## Options considered

1. Keep success values in the primary summary surfaces and rely on the detached
   manifest.
2. Compute cross-surface verdicts in memory before writing `summary.json` and
   `summary.md`.
3. Rewrite summary and manifest generation as a schema-driven renderer.

## Decision

Choose option 2. The no-WING writer now computes the manifest cross-surface
contract verdict from the summary dictionary and Markdown text before writing
the summary artifacts.

## Why this won

It makes the operator-facing primary artifacts fail closed without changing the
live boundary: no console contact, no mixer mutation, no readiness closure, and
no live authority.

## Rejected alternatives

Option 1 leaves stale success signals in the primary summary surfaces. Option 3
is larger than this focused gate needs.

## Implementation plan

- Factor cross-surface checking into an in-memory helper.
- Reuse that helper for the detached artifact manifest.
- Apply the helper while writing the summary surfaces.
- Render computed mismatch and match fields into `summary.md`.
- Add a stale-digest regression test.

## Test plan

- Focused manifest cross-surface tests.
- no-WING quick check.
- Project Director guard.
- Standard pytest.

## Risks and rollback

Risk is limited to offline report rendering. Roll back by removing the
in-memory cross-surface summary check and restoring the previous static success
fields.
