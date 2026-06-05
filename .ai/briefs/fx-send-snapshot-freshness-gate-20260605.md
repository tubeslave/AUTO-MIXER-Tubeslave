# Brief: fx-send-snapshot-freshness-gate-20260605

## Context

`fx_send_proposal` can consume `fx_inserts_snapshot` schema evidence for single
and batch proposal flows. A mapped schema from a stale snapshot should not allow
an FX-send recommendation to become proposal-only, because the selected FX slot
semantics may no longer match the console state.

## Task

Block FX-send proposal promotion when schema evidence came from a stale
`fx_inserts_snapshot`.

## Constraints

- Keep `fx_send_proposal.py` transport-free.
- Do not import Product Layer builders.
- Do not query or mutate mixer state.
- Preserve `can_apply_live=False` and `can_import_to_queue=False`.
- Keep existing explicit `current_state_cache_freshness` behavior intact.

## Acceptance

- A mapped but stale `fx_inserts_snapshot` produces `analysis_only`.
- The blocker is visible in recommendation evidence and severity.
- Batch proposal building filters stale shared snapshots.
- Registry metadata records the snapshot freshness gate.
- Focused, broad, and full tests pass.
