# Archive policy

This folder is for legacy or experimental code that should be preserved but removed from the active runtime surface.

## Rules

1. Do not move files here until `docs/cleanup_inventory.md` explains why they are legacy.
2. Do not archive files that are imported by live code.
3. Do not archive safety, mixer, OSC, rollback, or live controller code without a passing test suite.
4. Do not delete archived files in the same PR that moves them.
5. Every archive subfolder should contain a README explaining:
   - original path;
   - reason for archival;
   - whether it was live/offline/experiment;
   - whether any useful ideas should be ported back;
   - date and branch of archival.

## Suggested subfolders

```text
archive/
  legacy/
  experiments/
  old_backends/
  old_frontends/
  branch_snapshots/
```

## Safe archival workflow

1. Mark file as `archive` candidate in `docs/cleanup_inventory.md`.
2. Confirm it is not imported by live startup.
3. Move to archive in a dedicated commit.
4. Add compatibility note or replacement path.
5. Run compile/tests.
6. Only delete in a later cleanup after review.
