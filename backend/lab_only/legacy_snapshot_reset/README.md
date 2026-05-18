LAB-ONLY legacy snapshot/reset scripts

These scripts perform direct WING snapshot loads or broad reset-style writes and are not part of the approved supervised pilot runtime.

Use them only for explicit lab workflows from the repo root, for example:

`python3 -m backend.lab_only.legacy_snapshot_reset.load_snap_final <IP> <SNAP_NAME_OR_INDEX>`

`python3 -m backend.lab_only.legacy_snapshot_reset.reset_modules_trim_faders <IP> <PORT> --yes`

The approved WING deployment path remains the supervised backend/runtime flow described in:

- `Docs/reports/tub/TUB-327-Legacy-Runtime-Quarantine-Matrix.md`
- `Docs/reports/tub/TUB-364-Analyzer-Control-Runtime-Consolidation.md`
- `Docs/RUNTIME_HYGIENE.md`
