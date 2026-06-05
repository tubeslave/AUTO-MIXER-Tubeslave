# Brief: fx-send-batch-snapshot-bridge-20260605

## Context

`fx_send_proposal` can derive scoped FX parameter schema evidence from a
read-only `fx_inserts_snapshot` for a single analysis input. Real assisted or
autonomous preparation flows usually build FX-send recommendations for multiple
channels from one current Product Layer snapshot.

## Task

Allow `build_fx_send_proposals()` to accept one shared
`current_fx_inserts_snapshot` and propagate it to analyses that do not already
carry explicit schema or snapshot evidence.

## Constraints

- Keep the existing call shape backward-compatible.
- Do not mutate input dataclass instances.
- Do not import Product Layer builders.
- Do not query or mutate mixer state.
- Keep proposals non-queueable and non-live-applyable.

## Acceptance

- Batch proposal building can use one mapped snapshot schema for multiple
  channels.
- A generic shared snapshot filters affected FX-send items out of proposal-only
  results.
- Registry metadata records the batch snapshot bridge.
- Focused, broad, and full tests pass.
