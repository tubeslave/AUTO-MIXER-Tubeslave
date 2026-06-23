# ADR: WING MVP Review Template Strict Scope Type Gate

## 1. Context

The WING read-only review decision template builder copied `channels` and
`operator_confirmed_channels` from report rows when those values were lists.
That allowed malformed report mappings such as `["3"]` or `[3.0]` to reach
template candidate checks.

## 2. Options Considered

- Keep relying on evidence intake and decision-ledger validation.
- Add broad schema validation to all review reports.
- Add local non-coercing integer-list normalization in the review template tool.

## 3. Decision

Add a local integer-list helper and use it before review row output and template
candidate validation.

## 4. Why This Won

The decision template is a manual-review boundary and should fail closed on its
own. The change is dependency-free, narrow, and preserves no-live-authority
semantics.

## 5. Rejected Alternatives

Relying on intake alone is weaker because the function accepts report mappings.
Broad report schema validation is larger than needed for this channel-scope
gate.

## 6. Implementation Plan

- Add `_int_list()` to the review tool.
- Use it for `channels` and `operator_confirmed_channels` in review rows.
- Use it when building the manual review decision template.
- Add a regression test for string and float channel scope values.

## 7. Test Plan

Run focused WING review/evidence/no-WING tests and the standard repository
pytest command.

## 8. Risks And Rollback

Risk is limited to rejecting malformed review data. Rollback is reverting the
helper and regression test if a real report format unexpectedly emits
non-integer channel scope values, which would require operator review before
live use.
