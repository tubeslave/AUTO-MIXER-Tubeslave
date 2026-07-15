# ADR: M119 Private Artifact Root Verifier CLI

## 1. Context

M111 durable evidence must live below a private non-group-writable runtime
root. The migration builder accepts that root, and the verifier API accepts it,
but the `--verify-summary` CLI branch discarded the parsed `--artifact-root`.
Consequently the documented verifier falsely rejected a valid private-root
summary as outside the repository-default root.

Kimi CLI returned HTTP 402 in read-only plan mode. Independent read-only Codex
reviewers supplied the fallback proposal and confinement review.

## 2. Options considered

- Forward the explicit root to the existing verifier.
- Infer the root from the summary parent.
- Add another verifier-only root option.
- Relax durable evidence-store directory permissions.

## 3. Decision

Forward `args.artifact_root` unchanged to `verify_migration_paths()` when
`--verify-summary` is selected. Preserve lexical paths while checking every
root/run directory component for symlinks, and require the resolved run
directory to be a direct child of the resolved root. Keep nested-artifact
confinement and no-authority checks.

## 4. Why this won

It restores build/verify symmetry while keeping an independently selected
trust boundary. A broad ancestor or symlink alias cannot authorize the run,
the summary cannot authorize its own parent, and no live policy changes.

## 5. Rejected alternatives

Parent inference removes meaningful confinement. A duplicate option adds
ambiguity. Relaxing store permissions weakens durable evidence integrity.

## 6. Implementation plan

Patch the CLI call and confinement checks, add exact forwarding plus
inside/outside/broad/symlink regressions, validate, commit, rebind CP05, then
recreate the source-bound challenge and sealed replay.

## 7. Test plan

Run focused CLI/direct-child/symlink tests, all M005 migration tests, WING
rails, full repository tests and a fresh private-root M111 replay plus CLI
verification.

## 8. Risks and rollback

An incorrect or overly broad root could falsely widen accepted paths; strict
direct-child and symlink tests prevent this. Revert M119 and its paired CP05 marker together,
and never reuse a challenge from either source identity.
