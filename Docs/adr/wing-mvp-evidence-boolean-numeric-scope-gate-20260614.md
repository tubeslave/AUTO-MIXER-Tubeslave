# ADR: WING MVP Evidence Boolean Numeric Scope Gate

## 1. Context

The WING read-only preflight artifact validator uses Python helpers to normalize
numeric fields. Python booleans are subclasses of integers, so a malformed JSON
artifact could otherwise present `true` where a channel id or count is expected.

## 2. Options Considered

- Keep current coercion and rely on later mismatches.
- Add a broad schema dependency.
- Reject booleans inside the existing numeric helpers.

## 3. Decision

Reject booleans inside the existing read-only evidence numeric helpers.

## 4. Why This Won

It protects the candidate boundary directly, keeps the diff small, avoids new
dependencies, and preserves all existing no-live-authority behavior.

## 5. Rejected Alternatives

Keeping the current coercion leaves an avoidable ambiguity in safety-critical
artifact intake. A new schema dependency is unnecessary for this narrow gate.

## 6. Implementation Plan

- Update `_safe_int()` to return `0` for boolean inputs.
- Update `_int_tuple()` to skip boolean list entries.
- Add a regression test that proves boolean scope/count inputs reject the
  preflight candidate.

## 7. Test Plan

Run the focused WING evidence/review/no-WING tests and the standard repository
pytest command.

## 8. Risks And Rollback

Risk is limited to rejecting malformed artifacts that previously might have
been coerced. Rollback is reverting the helper and regression test if a real
artifact format unexpectedly emits booleans, which would itself need operator
review before any live use.
