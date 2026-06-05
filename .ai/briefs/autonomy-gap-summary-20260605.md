# autonomy-gap-summary-20260605

## Context

`autonomy_promotion_readiness` already exposes one row per promotion target.
Operators and future agents can see blockers on each row, but there is no
compact Product Layer summary that groups the remaining autonomy gaps by
category or action family.

## Goal

Add a read-only gap summary to the existing autonomy promotion readiness
snapshot. The summary should make it clear which targets are blocked by safety
proof, read coverage, write-surface quarantine, capability gaps, or Full
Autonomous mode policy.

Expose that summary in the existing frontend Settings module status table as
read-only `promotion_gap:*` rows so operators can see the gap buckets without
new controls.

## Non-goals

- Do not change operator mode policy.
- Do not create or queue console actions.
- Do not widen `supervised_live_allowed` or live write kinds.
- Do not add a new websocket route.
- Do not add frontend apply buttons.

## Safety constraints

- The new summary must be derived only from existing readiness rows.
- The snapshot must continue to report `live_write=false`.
- Full Autonomous mode must remain blocked.

## Expected evidence

- Snapshot summary contains deterministic gap bucket counts.
- Scene/preset and routing remain visibly blocked.
- Evidence requirement IDs are grouped by action family.
- Frontend adapter emits read-only promotion-gap status rows.
- Focused readiness tests pass.
