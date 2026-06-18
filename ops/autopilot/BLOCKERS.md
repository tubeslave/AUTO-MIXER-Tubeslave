# Autopilot Blockers

## Current blockers

- M001 is waiting for external validation: real WING Rack read-only artifact
  and operator review are required before physical readback intake can begin.
- Real WING Rack hardware evidence is not available from this setup run.
- Existing worktree was dirty before setup, with unrelated modified backend,
  tests, docs and generated artifacts. Autopilot must not overwrite them.
- Future heartbeat wake-ups must verify native Goal transition tools before
  replacing a thread Goal.

## Not blockers

- `ops/autopilot/STOP` does not exist after setup.
- Control-plane docs/state can proceed without live hardware contact.
