# ADR: WING MVP Review Template Expected Artifact Path Gate

## 1. Context

The WING read-only review decision template builder required non-empty
`summary_path` and `events_path` fields, but did not verify that they matched
the expected summary and event artifact paths surfaced by evidence intake.

## 2. Options Considered

- Keep relying on evidence intake.
- Add broad schema validation to all review reports.
- Add local expected-artifact path validation in the review template tool.

## 3. Decision

Require summary and event paths to match their expected artifact paths before
template generation.

## 4. Why This Won

The decision template is a manual-review boundary and should fail closed on its
own. The change is narrow, dependency-free, and preserves no-live-authority
semantics.

## 5. Rejected Alternatives

Relying on intake alone is weaker because the function accepts report mappings.
Broad report schema validation is larger than needed for this path-binding
gate.

## 6. Implementation Plan

- Read expected summary and event artifact fields from the candidate row.
- Require them to equal the actual summary and event paths.
- Require the evidence match flags to be true.
- Add a regression test for false match flags and mismatched expected paths.

## 7. Test Plan

Run focused WING review/evidence/no-WING tests and the standard repository
pytest command.

## 8. Risks And Rollback

Risk is limited to rejecting malformed review data. Rollback is reverting the
path checks and regression test if a real report format unexpectedly omits
expected artifact bindings, which would require operator review before live use.
