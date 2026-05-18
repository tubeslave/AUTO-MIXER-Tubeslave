# Legacy Backend Backups

Date: 2026-05-18
Status: quarantine

This folder stores old backup files that should not sit next to active runtime
modules.

Current contents:

- `server.py.bak` - historical backend server backup. Do not import or run this
  as a production server.

If a backup contains useful behavior, extract the behavior into a reviewed
patch and tests instead of restoring the backup wholesale.
