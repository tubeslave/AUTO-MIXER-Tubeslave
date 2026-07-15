# M119 Private Artifact Root Verifier CLI

## Task

Make `--verify-summary` honor the same explicit `--artifact-root` accepted by
the M005 migration builder, without weakening confinement or live-safety.

## Evidence

- A clean M111 replay under `~/.automixer/runtime` passed every replay, writer,
  socket and audio-I/O check.
- `verify_migration_paths(..., artifact_root=<private root>)` accepts summary
  confinement and leaves only the missing console-attestation checks.
- The CLI parses `--artifact-root` but calls `verify_migration_paths()` without
  it, so the same summary is falsely treated as outside the repository-default
  artifact root.

## Required outcome

1. Forward the explicit CLI root to the verifier.
2. Keep that root as the caller-selected confinement boundary.
3. Require a real, non-symlink root and run-directory chain.
4. Require the run directory to be a direct child of the root, matching the
   builder contract.
5. Add positive, wrong-root, broad-root and symlink CLI regressions.
6. Do not contact backend transport, audio devices or WING.
