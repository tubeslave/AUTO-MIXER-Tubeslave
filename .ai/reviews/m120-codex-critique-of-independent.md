# Codex critique of independent M120 proposal

## 1. Strongest part of the other proposal

The independent diagnosis precisely identifies the real ordering contract:
`connection_status/connected=false` is followed by a truthful readiness
invalidation. It correctly refuses to delete that backend broadcast or accept
the route artifact without verified teardown. Its negative test requiring a
real disconnect acknowledgement is essential.

## 2. Possible mistakes or blind spots

Adding an opt-in to the generic receiver makes a safety exception available to
every future call site. A default-false flag limits immediate risk but does not
encode that the exception is valid only for terminal disconnect. The proposal
also needs an exact payload envelope (`running=false`, and when present
`is_running=false`); accepting only type/status is broader than the observed
contract.

## 3. Where the solution is too complex, fragile, or costly

The opt-in is mechanically small, but it adds conditional behavior to both the
normal receive loop and duplicate-grace loop. That makes active control
terminal reasoning more fragile than a dedicated teardown receiver whose name
and call graph prove the exception cannot be used for Apply, Proof or Audit.

## 4. What should be kept

- Keep the backend broadcast unchanged.
- Require a real `connected=false` acknowledgement.
- Preserve the existing active-session invalidation test unchanged.
- Add the real two-message backend sequence to regression coverage.

## 5. Revised recommendation

Use a private `_receive_disconnect_terminal()` rather than a generic opt-in.
It should allow one exact readiness-invalidation notification before or after
the disconnect acknowledgement, require exactly one valid disconnect terminal,
reject duplicates and all unrelated failures, and finish only after the same
short duplicate-grace interval. This is a few more lines but gives the safer
API boundary.
