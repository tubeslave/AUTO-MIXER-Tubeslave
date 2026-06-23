# ADR: WING MVP no-WING Artifact Manifest Summary Surface Self-Check

## 1. Context

The no-WING summaries expose the detached artifact manifest contract. Operators
need assurance that the markdown surface renders the same required manifest
fields that automation expects.

## 2. Options considered

- Keep JSON-only manifest surface assertions.
- Add a markdown field-line self-check.
- Add a separate sidecar schema validator.

## 3. Decision

Add a fail-closed markdown field-line self-check for the artifact manifest
summary surface.

## 4. Why this won

It follows the existing proof-plan operator markdown self-check pattern and keeps
the no-WING handoff reviewable from the primary operator summary.

## 5. Rejected alternatives

JSON-only checks were rejected because they do not prove the operator-readable
summary renders the fields. A separate schema validator was rejected as too large
for this no-WING handoff slice.

## 6. Implementation plan

- Add required manifest summary field-line constants.
- Add a helper that records rendered/missing status and fails closed on missing
  lines.
- Add tests and Project Director baseline markers.

## 7. Test plan

Run focused no-WING summary tests, Project Director/WING rail tests, full pytest,
`git diff --check`, and memory lint.

## 8. Risks and rollback

Risk is limited to marker drift. Roll back by removing the self-check fields,
helper, and matching baseline markers. This ADR does not contact WING, grant live
authority, or authorize fader/gain movement.
