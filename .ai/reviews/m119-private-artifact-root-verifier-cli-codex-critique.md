# Codex Critique of Independent Proposal: M119

## 1. Strongest part of the other proposal

It keeps the caller-provided root as the trust boundary and explicitly rejects
deriving authority from the summary path itself.

## 2. Possible mistakes or blind spots

A forwarding-only mock test would not prove the real verifier still rejects an
outside-root summary. A simple descendant check also accepts an overly broad
ancestor, and eager path resolution hides symlink aliases.

## 3. Where the solution is too complex, fragile, or costly

A new verifier-specific option would duplicate an existing concept and create
two roots whose precedence could drift.

## 4. What should be kept from the other proposal

Keep exact forwarding, explicit boundary semantics, direct-child parity with
the builder, symlink rejection and negative confinement regressions.

## 5. My revised recommendation

Make the plumbing fix, harden root/run confinement, and add the full cheap CLI
matrix. Do not widen downstream M006 authority or evidence-store permissions.
