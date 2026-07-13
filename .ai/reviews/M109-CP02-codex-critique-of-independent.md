# Codex critique of the independent CP02 proposal

## 1. Strongest part

The independent proposal specifies an end-to-end review boundary rather than a
list truncation patch: immutable content, exact byte caps, chunked detail and
pre-mutation digest enforcement belong in one vertical slice.

## 2. Possible mistakes or blind spots

- Requiring producers to label global hazards is correct, but existing rows
  without that enum need an explicit conservative migration state.
- A frozen snapshot store needs hard object, byte and per-connection limits or
  it can become a new memory exhaustion surface.
- Full `raw` is safest for integrity, but canonicalization must fail closed on
  NaN, bytes and other non-JSON values with a typed creation error.

## 3. Where it is too complex, fragile or costly

Snapshot pagination plus chunked detail is more state than a simple truncated
list. The complexity is justified only if implemented in an isolated module
with pure packing/canonicalization functions and deterministic tests.

## 4. What should be kept

Keep the 131,072-byte invariant, 65,536-byte raw chunks, server-side version
selection, authenticated detail, no legacy apply, and zero-side-effect binding
errors.

## 5. Revised recommendation

Ship the vertical slice atomically, but stage implementation as pure server
contract, handler integration, then active frontend. Do not widen WING authority
or combine this work with musical-correction policy.
