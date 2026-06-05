# Brief: fx-send-snapshot-schema-bridge-20260605

## Context

The FX send proposal layer can now block recommendations when explicit FX
parameter schema evidence is generic, missing, or mismatched. The Product Layer
already exposes that evidence inside `fx_inserts_snapshot`, but callers would
otherwise have to manually extract the right slot schema before invoking
`fx_send_proposal`.

## Task

Bridge `fx_send_proposal` to read-only `fx_inserts_snapshot` dictionaries so the
proposal layer can derive scoped FX parameter schema evidence without importing
Product Layer builders or querying hardware.

## Constraints

- Keep `fx_send_proposal.py` transport-free.
- Do not import `operator_product_state`.
- Do not query or mutate mixer state.
- Do not make FX proposals queueable or live-applyable.
- Preserve fallback compatibility when no snapshot is supplied.

## Acceptance

- `FXSendAnalysisInput` accepts an optional `current_fx_inserts_snapshot`.
- A mapped slot schema matching the selected FX family can satisfy the schema
  gate.
- Generic/unknown snapshot schema keeps recommendations `analysis_only`.
- Registry metadata records the snapshot-schema bridge.
- Focused, broad, and full tests pass.
