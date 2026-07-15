# Independent Critique of Codex Proposal: M119

Independent read-only Codex reviewer fallback; this is not a Kimi opinion.

## 1. Strongest part of the other proposal

It fixes build/verify asymmetry without changing the verifier's path or
symlink rules.

## 2. Possible mistakes or blind spots

The CLI will still return blocked for an incomplete migration. The regression
must assert the specific confinement check, not require overall success. It
must also reject a broad ancestor and symlinked root/run directories.

## 3. Where the solution is too complex, fragile, or costly

The implementation is already minimal. Avoid changing result schemas or
special-casing the private home directory.

## 4. What should be kept from the other proposal

Keep the explicit root, positive and negative real checks, and the fresh
source-bound replay requirement after commit.

## 5. My revised recommendation

Accept forwarding only together with direct-child and no-symlink confinement,
the expanded CLI matrix, and unchanged no-authority flags.
