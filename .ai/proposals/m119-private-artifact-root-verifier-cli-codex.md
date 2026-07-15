# Codex Proposal: M119 Private Artifact Root Verifier CLI

## 1. Short thesis

Forward the already parsed `--artifact-root` into `verify_migration_paths()` in
verify mode and lock that behavior with CLI-level confinement tests.

## 2. What I understood

The verifier API already supports an explicit root and correctly rejects a
summary outside it. Only the CLI plumbing drops the value. This makes the
private runtime root required by the durable evidence store impossible to
verify through the documented CLI.

## 3. Proposed solution

- Call `verify_migration_paths(args.verify_summary,
  artifact_root=args.artifact_root)`.
- Preserve lexical paths long enough to reject symlink components.
- Require the resolved run directory to be a direct child of the resolved
  explicit root, matching the builder.
- Test exact argument forwarding.
- Test that a regular summary below the explicit root passes the confinement
  check even when other expected migration evidence is absent.
- Test that the same summary fails confinement under a different root.

## 4. Likely files to touch

- `tools/migrate_m005_current_release_provenance.py`
- `tests/test_migrate_m005_current_release_provenance.py`
- M119 council, ADR, plan and truth records

## 5. Alternatives considered

- Infer the root from the summary path: rejected because it makes every chosen
  summary self-authorizing.
- Weaken the durable-store ancestor permissions: rejected because it reduces
  artifact integrity.
- Move the private replay into the repository: rejected because the checkout
  is group-writable and correctly rejected by the evidence store.

## 6. Risks

The explicit root is a trust boundary selected by the operator. A broad
ancestor or symlinked root must not relabel an arbitrary tree as confined. The
verifier must also prove every nested artifact remains inside the run directory.

## 7. Test plan

Run forwarding, direct-child, broad-root and symlink regressions, the M005
migration suites, WING truth rails and
the standard repository suite. After commit/rebind, create a fresh challenge
and repeat the clean sealed replay below the private root.

## 8. Where the other agent may disagree

A reviewer may prefer a separate required verifier-only option. Reusing the
existing `--artifact-root` is less surprising and preserves build/verify
symmetry.
