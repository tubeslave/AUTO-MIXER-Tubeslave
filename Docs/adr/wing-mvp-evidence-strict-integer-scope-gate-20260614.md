# ADR: WING MVP Evidence Strict Integer Scope Gate

## 1. Context

The WING read-only preflight validator rejected boolean channel scope values,
but its list helper still coerced numeric strings and floats with `int(...)`.
For live-sound evidence intake, declared WING channel scope should be a strict
integer list.

## 2. Options Considered

- Keep coercion.
- Add a full JSON schema dependency.
- Reject non-integer list items in the existing helper.

## 3. Decision

Reject non-integer channel scope values in the existing helper.

## 4. Why This Won

It makes the preflight candidate boundary stricter without changing runtime
behavior, adding dependencies, or granting any live authority.

## 5. Rejected Alternatives

Coercion is too permissive for safety-critical artifact scope. A full schema
dependency is larger than needed for this narrow gate.

## 6. Implementation Plan

- Update `_int_tuple()` to accept only `int` values and continue excluding
  `bool`.
- Add a regression test for string and float channel scope fields.

## 7. Test Plan

Run the focused WING evidence/review/no-WING tests and the standard repository
pytest command.

## 8. Risks And Rollback

Risk is limited to rejecting malformed artifacts earlier. Rollback is reverting
the helper and regression test if a real capture unexpectedly emits non-integer
scope values, which would need operator review before live use.
