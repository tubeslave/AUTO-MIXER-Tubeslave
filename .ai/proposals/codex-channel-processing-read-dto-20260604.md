# Codex Proposal: Channel Processing Read DTO

Task id: channel-processing-read-dto-20260604
Date: 2026-06-04

## 1. Short thesis

Expose channel processing readback as a Product Layer snapshot before widening
any EQ, compressor, gate, or HPF automation. Cached observability is a
prerequisite, not live authority.

## 2. What I understood

The repository already has WING cached addresses and helper methods for EQ and
compression. Some methods can actively query the console, so the Product Layer
must avoid those paths in this step and only read existing cache.

## 3. Proposed solution

Add `build_channel_processing_snapshot` with:

- one row per channel;
- HPF, EQ bands, compressor, and gate sections;
- per-section `complete`, `partial`, or `missing` coverage;
- `mixer_state_cache` source metadata when cached values exist;
- safety flags: `read_only=true`, `live_write=false`,
  `mutates_console_state=false`, and `queries_console=false`;
- a websocket handler `get_channel_processing_snapshot`.

## 4. Likely files to touch

- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`

## 5. Alternatives considered

- Call `WingClient.get_channel_settings`.
- Fold processing fields into `channel_inventory`.
- Implement proposal/apply for EQ and dynamics now.

## 6. Risks

Cached state can be stale or incomplete. The snapshot must not claim physical
readback proof, and EQ/dynamics/gate writes must remain blocked outside the
approved supervised fader/gain path.

## 7. Test plan

- Test full cached processing coverage without `send`.
- Test missing cache coverage without query attempts.
- Test websocket handler round-trip.
- Run focused, broad, smoke, and full pytest validation.

## 8. Where the other agent may disagree

Kimi may prefer physical query fixtures first. That is valid follow-up work,
but a cached Product Layer DTO gives the fixture work a stable target schema.
