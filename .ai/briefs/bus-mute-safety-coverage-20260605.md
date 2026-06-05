# bus-mute-safety-coverage-20260605

## Context

Autonomy promotion readiness currently reports `safety_coverage_row_missing`
for bus/group/matrix and mute promotion targets. That hides the exact proof
contract required before these broad console controls can ever be reviewed for
supervised automation.

## Goal

Add fail-closed safety coverage rows for bus/group/matrix and mute action
families, then wire promotion readiness to those rows without granting live
authority.

## Non-goals

- Do not enable bus, DCA, matrix, master, or mute writes.
- Do not add mixer transport calls or physical WING proof.
- Do not reclassify read-surface coverage as complete.

## Safety constraints

- Both rows must keep `live_allowed=false`.
- Both rows must remain blocked/quarantined until readback, rollback,
  emergency-stop, fake-mixer, and real-WING proof exist.
- Existing supervised apply scope remains fader/gain only.

## Expected evidence

- Promotion readiness rows reference concrete safety row IDs instead of missing
  safety coverage.
- Bus/group/matrix and mute remain not supervised-live-allowed.
- Gap summaries still report safety/proof/read/write blockers and no Full
  Autonomous readiness.
