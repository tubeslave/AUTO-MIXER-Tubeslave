# ADR: WING MVP no-WING Artifact Manifest Self-Consistency Summary Surface

## 1. Context

The no-WING release check writes `summary.json`, `summary.md`, and
`artifact-manifest.json`. The manifest now computes self-consistency verdicts for
declared artifacts, SHA fields, and safety-denial flags. The operator handoff
entrypoint is still the summary pair.

## 2. Options considered

- Keep self-consistency only in `artifact-manifest.json`.
- Render the same verdict fields in `summary.json` and `summary.md`.
- Add a separate manifest-verdict report artifact.

## 3. Decision

Render the manifest self-consistency verdict fields in the no-WING summary JSON
and Markdown surfaces.

## 4. Why this won

The summary artifacts are the first operator-readable handoff surface. Showing
the sidecar verdict there reduces review friction while preserving the detached
manifest as the SHA-bound artifact.

## 5. Rejected alternatives

Keeping the verdict only in the sidecar hides an important bundle-integrity check
from the handoff entrypoint. A separate artifact would duplicate state and add
another file to validate.

## 6. Implementation plan

- Define the manifest self-consistency summary field set.
- Add the fields to the required JSON and Markdown summary surfaces.
- Render the values in `summary.md`.
- Update tests to prove summary and manifest agreement.

## 7. Test plan

- Targeted no-WING manifest summary surface tests.
- no-WING quick artifact inspection.
- Focused WING/no-WING/Director tests.
- Full standard pytest.

## 8. Risks and rollback

Risk: generated summary consumers may need to ignore additional no-WING fields.
Rollback: remove the summary surface fields and tests. The rollback does not
touch WING, mixer state, readiness closure, or live authority.
