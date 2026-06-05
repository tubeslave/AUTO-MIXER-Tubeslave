# frontend-promotion-readiness-status-20260605

## Context

The backend Product Layer exposes `autonomy_promotion_readiness` rows with
blockers and evidence requirement IDs, but the frontend Settings module status
adapter only shows method, write-surface, and safety-coverage rows. Operators
cannot see why broad targets like scene/preset and routing remain blocked from
the UI status surface.

## Goal

Map promotion readiness rows into the existing read-only Settings module status
table. The rows should remain status-only and must not add apply controls or live
write authority.

## Non-goals

- Do not create a new UI page.
- Do not add apply buttons, queue actions, or console writes.
- Do not treat `supervised_live_allowed` as a row-level executor permission in
  the module status table.

## Safety constraints

- Readiness rows must display `canWriteLive=false`.
- Evidence requirement IDs may be shown as next-step context only.
- Scene/preset and routing readiness must remain visibly blocked.

## Expected evidence

- Frontend adapter emits `promotion_readiness:*` rows.
- Scene/preset row includes evidence requirement IDs in searchable next-step
  text.
- Existing module/write/safety rows remain intact.
- Frontend contract tests pass.
