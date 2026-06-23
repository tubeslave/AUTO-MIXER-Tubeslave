# ADR: WING MVP Review Template Count Type Gate

## 1. Context

The WING read-only review decision template builder used `int(...)` for
`readback_count` and `physical_readback_count`. That could coerce malformed
values such as `true` or `"3"` into accepted counts.

## 2. Options Considered

- Keep relying on evidence intake.
- Add broad schema validation to review reports.
- Add a local non-coercing integer helper in the template tool.

## 3. Decision

Add a local non-coercing integer helper and use it before template generation.

## 4. Why This Won

The decision template is a manual-review boundary and should fail closed on its
own. The change is narrow, dependency-free, and preserves no-live-authority
semantics.

## 5. Rejected Alternatives

Relying on intake alone is weaker because the function accepts report mappings.
Broad schema validation is larger than needed for this count-type gate.

## 6. Implementation Plan

- Add `_non_bool_int()` to the review tool.
- Replace `int(...)` count normalization in review rows, template generation,
  and markdown rendering.
- Add a regression test for boolean and string count values.

## 7. Test Plan

Run focused WING review/evidence/no-WING tests and the standard repository
pytest command.

## 8. Risks And Rollback

Risk is limited to rejecting malformed review data. Rollback is reverting the
helper and regression test if a real report format unexpectedly emits
non-integer counts, which would require operator review before live use.
