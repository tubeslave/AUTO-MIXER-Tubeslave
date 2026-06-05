# Proposal: fx-inserts-read-dto-20260605

## Short thesis

Add a cached-only Product Layer DTO for FX rack slots and insert assignments,
keeping all FX/insert writes quarantined.

## What I understood

The autonomous goal needs FX management and insert awareness, but current safety
boundaries allow only supervised fader/gain live apply. FX slots and inserts can
change the processing chain, so the next safe step is read coverage.

## Proposed solution

Implement `build_fx_inserts_snapshot(server)` in
`backend/operator_product_state.py`. It reads `/fx/*` and insert addresses from
`mixer_client.state`, exposes coverage flags, and marks FX apply and insert
assignment as blocked. Register `get_fx_inserts_snapshot` in product-state
handlers and update the read surface inventory to Product snapshot coverage.

## Likely files to touch

- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`
- `tests/test_read_surface_inventory.py`

## Alternatives considered

- Use `get_fx_overview()` in the builder. Rejected because Product Layer
  builders should not call helpers that may query or scan hardware.
- Promote FX proposals to queue/apply. Rejected because readback, schema, and
  rollback proof are incomplete.

## Risks

- Cached FX parameters are raw numeric slots without semantic names.
- Cached insert assignments can be stale until physical-query fixture coverage
  exists.

## Test plan

- Unit test populated cached FX and insert state without transport calls.
- Unit test missing cache behavior.
- Handler round-trip and display-only contract tests.
- Full pytest suite and diff check.

## Where the other agent may disagree

Kimi may prefer waiting for FX parameter schemas first. My counterpoint is that
the DTO explicitly publishes raw/cache coverage and does not imply apply
authority.
