# ADR: WING MVP Review Template Value Type Gate

## 1. Context

The WING read-only review decision template builder copied any list-shaped
`value_types` field into pending decision templates. That let malformed report
mappings such as `["fader", 3]`, `["fader", "eq"]`, or duplicate lists reach
template candidate checks.

## 2. Options Considered

- Keep relying on evidence intake and review-decision ledger checks.
- Add broad schema validation to all review reports.
- Add local supported value-type validation in the review template tool.

## 3. Decision

Add local duplicate-free supported value-type validation and use it before
review row output and template candidate validation.

## 4. Why This Won

The decision template is a manual-review boundary and should fail closed on its
own. The change uses the shared contract, adds no dependency, and preserves
no-live-authority semantics.

## 5. Rejected Alternatives

Relying on intake alone is weaker because the function accepts report mappings.
Broad report schema validation is larger than needed for this value-type gate.

## 6. Implementation Plan

- Import the supported WING read-only value types from the shared contract.
- Add `_supported_value_type_list()` to the review tool.
- Use it for `value_types` in review rows and template generation.
- Add a regression test for non-string, unsupported, and duplicate value types.

## 7. Test Plan

Run focused WING review/evidence/no-WING tests and the standard repository
pytest command.

## 8. Risks And Rollback

Risk is limited to rejecting malformed review data. Rollback is reverting the
helper and regression test if a real report format unexpectedly emits
unsupported value types, which would require operator review before live use.
