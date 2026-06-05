# no-wing-read-surface-next-target-summary-20260605

## Context

Settings and Product Layer can display `next_physical_readback_surface_id`, but
the no-WING read-surface release summary still derives only the gap list. The
release artifact should carry the next target explicitly so downstream surfaces
do not infer it differently.

## Goal

Add `next_physical_readback_surface_id` to read-surface fixture evidence
snapshots and no-WING summaries, and validate that it matches the first physical
readback gap.

## Non-goals

- Do not contact WING or dLive hardware.
- Do not execute physical readback proof.
- Do not unlock Full Autonomous mode.

## Safety constraints

- The field is read-only planning evidence.
- The no-WING release check must fail if the target drifts from the gap list.
- `full_autonomous_ready` remains false while physical readback gaps exist.

## Expected evidence

- Snapshot summary includes the next physical readback surface ID.
- no-WING schema rejects target/list drift.
- no-WING `summary.json` and `summary.md` expose the field.
