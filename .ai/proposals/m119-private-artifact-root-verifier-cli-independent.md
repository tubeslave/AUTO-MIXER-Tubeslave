# Independent Proposal: M119 Private Artifact Root Verifier CLI

Independent read-only Codex reviewer fallback; this is not a Kimi opinion.

## 1. Short thesis

Treat the explicit CLI root as the confinement boundary and pass it unchanged
to the already strict verifier.

## 2. What I understood

The API behavior is correct. The failure is limited to one CLI branch that
silently falls back to the repository-relative default.

## 3. Proposed solution

- Forward both the summary path and explicit root.
- Do not derive the root from the summary.
- Require a real non-symlink root and run directory.
- Require the run directory to be the root's direct child.
- Cover broad ancestors and symlink aliases in addition to positive and
  negative confinement checks.

## 4. Likely files to touch

- Migration CLI and its unit tests
- Council/ADR/plan and current truth surfaces

## 5. Alternatives considered

- Automatic parent inference.
- A second verifier-specific root flag.
- Repository-local evidence only.

## 6. Risks

Automatic inference would erase the trust boundary. Resolving paths before
checking their lexical components would also hide root/run symlinks.

## 7. Test plan

Verify exact forwarding, direct-child acceptance, outside/broad-root rejection
and root/run symlink rejection; then rerun migration and WING safety suites.

## 8. Where the other agent may disagree

Codex may consider the forwarding test sufficient. A real verifier check is
also valuable because it proves the root changes confinement semantics without
weakening them.
