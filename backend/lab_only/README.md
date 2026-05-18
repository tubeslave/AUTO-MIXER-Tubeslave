# Backend Lab Only

This folder is for backend scripts that are legacy, experimental, manual, or
quarantined.

Rules:

- Do not import from this folder in live runtime.
- Do not use scripts here as release proof.
- If a script becomes product behavior, move the behavior into normal backend
  modules and add focused tests under `tests/`.
- If a script can write to WING, treat it as unsafe unless it is proven to use
  the supervised gate.

Current groups:

- `legacy_snapshot_reset/` - old snapshot/reset tools.
- `legacy_backups/` - old backup files preserved for audit only.
