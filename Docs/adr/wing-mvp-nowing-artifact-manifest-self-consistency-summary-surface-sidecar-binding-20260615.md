# ADR: WING MVP no-WING Artifact Manifest Self-Consistency Summary Surface Sidecar Binding

## 1. Context

The no-WING release check writes `summary.json`, `summary.md`, and
`artifact-manifest.json`. The summaries now carry a value-gate verdict for the
manifest self-consistency summary surface. The detached manifest binds the
summary artifacts by path and SHA256.

## 2. Options considered

- Keep the value-gate verdict only in the summaries.
- Mirror the value-gate verdict into `artifact-manifest.json`.
- Add another detached value-gate artifact.

## 3. Decision

Mirror the value-gate verdict into `artifact-manifest.json`.

## 4. Why this won

The manifest is the detached bundle index. It should show whether the summary
surfaces it hashes also passed the value-gate check.

## 5. Rejected alternatives

Summary-only diagnostics leave the sidecar incomplete. A separate artifact would
add another file without adding new evidence.

## 6. Implementation plan

- Read generated `summary.json` and `summary.md` in the manifest writer.
- Recompute the manifest self-consistency summary-surface value gate.
- Add those fields to `artifact-manifest.json`.
- Extend happy-path manifest coverage.

## 7. Test plan

- Targeted no-WING manifest tests.
- no-WING quick artifact inspection.
- Focused WING/no-WING/Director tests.
- Full standard pytest.

## 8. Risks and rollback

Risk: generated manifest consumers may need to ignore additional diagnostic
fields. Rollback: remove the sidecar value-gate fields and tests. The rollback
does not touch WING, mixer state, readiness closure, or live authority.
