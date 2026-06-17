# ADR: WING MVP Project Director Selected-Action Summary JSON Surface

Date: 2026-06-15

## 1. Context

Project Director routes Goal-mode work from the current first command. The
selected-action anchor was available inside the no-WING handoff packet artifact
and rendered in `summary.md`, but `summary.json` did not expose a compact
top-level machine surface for selected-action and manifest-sidecar checklist
verification.

## 2. Options Considered

1. Keep the data nested only.
2. Add a separate selected-action artifact.
3. Project selected-action verdicts into top-level no-WING `summary.json`.

## 3. Decision

Project selected-action verdicts into top-level `summary.json`.

## 4. Why This Won

The no-WING summary is the release gate's machine-readable surface. It should
let consumers verify current-command source, digest, baseline coverage, and
sidecar checklist presence without parsing nested packet blobs.

## 5. Rejected Alternatives

Nested-only data was rejected because it is brittle for external checks. A new
artifact was rejected because the summary projection is enough and avoids
another file contract.

## 6. Implementation Plan

- Build a selected-action summary projection from the handoff artifact.
- Copy it into top-level no-WING `summary.json`.
- Add tests for the new fields.
- Register evidence and regenerate latest progress.

## 7. Test Plan

- Targeted no-WING summary test.
- no-WING quick.
- Focused WING/no-WING/Director tests.
- Project Director guard snapshot.
- Memory lint and `git diff --check`.
- Full standard pytest.

## 8. Risks and Rollback

Risk is limited to extra summary fields. Roll back by removing the projection,
test assertions, and evidence docs if the Project Director anchor contract is
replaced.
