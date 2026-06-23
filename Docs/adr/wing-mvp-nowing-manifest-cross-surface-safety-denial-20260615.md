# ADR: WING MVP no-WING Manifest Cross-Surface Safety Denial

## 1. Context

The no-WING release check writes `summary.json`, `summary.md`, and
`artifact-manifest.json` for the current WING read-only preflight handoff. These
artifacts must preserve the boundary that offline evidence cannot close
readiness or grant live authority.

## 2. Options considered

- Keep only contract id and digest cross-surface checks.
- Add exact safety-denial checks to the existing cross-surface contract.
- Add a separate post-generation artifact validator.

## 3. Decision

Add exact JSON and Markdown safety-denial verdicts to the existing manifest
cross-surface check.

## 4. Why this won

The existing check is already the authority for cross-surface manifest contract
diagnostics. Extending it keeps the operator-facing failure reason in the same
place and prevents a `true` readiness/live-authority line from passing merely
because the line exists.

## 5. Rejected alternatives

Keeping only digest checks was too weak for a live-sound safety boundary. A
separate validator would duplicate the same surface knowledge and increase the
chance of inconsistent diagnostics.

## 6. Implementation plan

- Add four computed safety-denial verdict fields.
- Include those fields in required JSON and Markdown summary surfaces.
- Render and replace their Markdown values after cross-surface validation.
- Add a regression test for unsafe `true` safety lines.

## 7. Test plan

- Targeted no-WING manifest cross-surface tests.
- Focused WING/no-WING/Director tests.
- Full standard pytest.

## 8. Risks and rollback

Risk: required summary fields drift and fail no-WING validation. Rollback:
remove the new verdict fields and their required-surface entries. This rollback
does not touch live mixer behavior.
