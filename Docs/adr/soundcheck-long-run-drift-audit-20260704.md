# ADR: Soundcheck Long-Run Drift Audit

## Context

The current executable slice allows read-only soundcheck runtime observation,
but it must not create proposals, apply corrections or widen WING authority
without the M005 supervised disconnect/timeout/stale-state proof and M006 owner
readiness review. Existing daemon logs include historical broad soundcheck
activity, so they need a repeatable offline check before anyone can treat them
as evidence.

## Options considered

1. Reuse the old daemon logs as-is.
2. Run a fresh live proposal/apply loop.
3. Add an offline audit that reads existing JSONL logs and fails closed when
   duration, target semantics, executor use or cumulative relative drift do not
   meet the contract.

## Decision

Add `tools/soundcheck_long_run_drift_audit.py` and targeted tests. The tool
reads JSONL evidence only, writes a summary artifact, and always denies live
authority and readiness closure in its output.

## Why this won

It creates a machine-checkable gate without contacting WING or invoking
`apply_operator_proposal`. It also turns unsafe old daemon evidence into a
clear negative artifact instead of leaving it ambiguous.

## Rejected alternatives

- Reusing the old daemon log directly was rejected because it contained live
  executor actions, relative-only targets and cumulative drift.
- Running a fresh live apply/proposal loop was rejected because unattended
  Codex work is limited to the current safe slice.

## Implementation plan

1. Parse one or more JSONL evidence files.
2. Count proposal/action rows from `applied`, `proposals`, `proposed`,
   `recommendations` and blocked decisions.
3. Fail when duration is below the configured minimum, actions are missing,
   live executor actions appear, fader boosts appear, gain/trim boosts appear,
   fader/trim/gain targets are relative-only, or cumulative relative drift
   exceeds the threshold.
4. Write the summary under `artifacts/soundcheck_long_run_drift_audit/`.
5. Inventory the tool as read-only proof tooling.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_soundcheck_long_run_drift_audit.py tests/test_write_surface_inventory_static_scan.py -q`
- Run the audit against `reports/soundcheck_daemon_log_20260703T221457.jsonl`
  and record the failed summary as negative evidence.

## Risks and rollback

The main risk is misclassifying EQ band gain as input gain/trim. The tests cover
that distinction. Rollback is to remove the tool, its tests, the inventory row
and the generated audit artifact; no live state is touched.
