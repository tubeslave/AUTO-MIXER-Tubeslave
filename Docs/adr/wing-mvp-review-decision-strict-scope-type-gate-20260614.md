# ADR: WING MVP Review Decision Strict Scope Type Gate

## 1. Context

The WING read-only review decision ledger parsed `channels` and
`operator_confirmed_channels` with `int(...)`. This could coerce malformed
manual decision values such as `"3"` or `3.0` into accepted integers.

## 2. Options Considered

- Keep coercion and rely on candidate comparison.
- Add broad schema validation.
- Make the existing ledger helper strict.

## 3. Decision

Make the ledger helper strict: accept only real `int` values and reject `bool`,
strings, and floats.

## 4. Why This Won

The ledger is the last manual-decision acceptance boundary before supervised
proof planning. It should fail closed independently and without new
dependencies.

## 5. Rejected Alternatives

Coercion is too permissive for safety-critical manual decisions. Broad schema
validation is larger than needed for this specific gate.

## 6. Implementation Plan

- Update `_int_list()` in the review decision ledger.
- Add a regression test for string/float channel scope in decision records.

## 7. Test Plan

Run focused WING evidence/review/no-WING tests and the standard repository
pytest command.

## 8. Risks And Rollback

Risk is limited to rejecting malformed decision records. Rollback is reverting
the helper and regression test if a legitimate decision format unexpectedly
emits non-integer scope, which would require operator review before live use.
