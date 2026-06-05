# autonomy-mode-gap-bridge-20260605

## Context

`autonomy_promotion_readiness.summary.promotion_gap_summary` now groups
promotion blockers into stable autonomy gap buckets. The mode transition gate
still reports Full Autonomous as blocked using broad global blockers, without
linking the mode decision back to those specific gap buckets.

## Goal

Expose readiness gap bucket counts and blocked target IDs through
`autonomy_mode_transition_gate` rows. Full Autonomous mode should list the gap
buckets as blockers while remaining read-only and blocked.

## Non-goals

- Do not unlock Full Autonomous mode.
- Do not change runtime operator mode.
- Do not alter supervised live write kinds.
- Do not create console actions.

## Safety constraints

- The gate must remain transport-free.
- `autonomous_apply_allowed` must stay false.
- `mode_expansion_allowed` must stay false.

## Expected evidence

- Mode gate rows include readiness gap bucket counts.
- Auto row includes readiness gap blockers.
- Supervised remains limited to existing fader/gain policy.
- Mode gate tests pass.
