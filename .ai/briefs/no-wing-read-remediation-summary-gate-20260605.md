# no-wing-read-remediation-summary-gate-20260605

## Context

The backend and cockpit now expose `read_surface_remediation_plan` with the next
physical readback action. The no-WING release gate still validates only
`read_surface_fixture_evidence`, so a drift between fixture target and
remediation action can escape release summaries.

## Goal

Add a no-WING release validation and summary section for read-surface
remediation plan consistency.

## Non-goals

- Do not execute remediation actions.
- Do not query WING or dLive.
- Do not change live write policy or Full Autonomous readiness.

## Safety constraints

- The validation must require read-only/non-mutating safety flags.
- The next remediation action must match the fixture evidence next physical
  readback target.
- Exactly one remediation action may be flagged as the next physical readback
  target while physical fixture gaps remain.
- Summary data must remain evidence only.

## Expected evidence

- `read_surface_remediation_plan_schema` appears in no-WING results.
- Summary JSON and Markdown include next physical readback action ID.
- Summary JSON and Markdown include the action IDs flagged as the next physical
  readback target.
- Tests reject mismatched remediation target/action data.
- Tests reject multiple next-target action flags.
