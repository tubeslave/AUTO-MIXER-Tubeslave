# Autopilot Operations

## Start

Use the `automixer-autopilot` skill from this checkout. The automation should be
a thread heartbeat attached to the current Codex thread with a 20 minute cadence.

## Stop

Create:

`ops/autopilot/STOP`

At the next safe checkpoint automation must stop new work, release lease, update
state/report/ledger and exit.

## Lease recovery

If `ops/autopilot/WRITER_LEASE/owner.json` heartbeat is older than 30 minutes
and the owning process/thread is not active, mark it stale in
`RUN_LEDGER.jsonl`, remove the lease directory and resume from the first
unverified checkpoint.

## Validation commands

- `python3 -m json.tool ops/autopilot/STATE.json`
- `python3 - <<'PY'` with JSONL validation for `ops/autopilot/RUN_LEDGER.jsonl`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Permissions

Default automation permissions are workspace-scoped, with no unattended network
or hardware contact. Real WING operations require operator setup and explicit
approval under the runbook.

