# Codex Proposal: Matrix Read DTO

Task id: matrix-read-dto-20260604
Date: 2026-06-04

## 1. Short thesis

Expose cached matrix state as a Product Layer DTO before any master/matrix
optimization can be treated as more than proposal-only analysis.

## 2. What I understood

WING has matrix EQ and insert cache paths, but existing Product Layer snapshots
do not normalize matrices. Some helper methods can actively query inserts, so
this step must read only `client.state`.

## 3. Proposed solution

Add `build_matrix_snapshot` with:

- one row per matrix;
- basic controls: fader, mute, pan;
- 8-band EQ cache;
- pre/post insert cache;
- per-section `complete`, `partial`, or `missing` coverage;
- safety flags: `read_only=true`, `live_write=false`,
  `mutates_console_state=false`, and `queries_console=false`;
- websocket handler `get_matrix_snapshot`.

## 4. Likely files to touch

- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`

## 5. Alternatives considered

- Keep matrix state as raw cache.
- Call `get_matrix_inserts`.
- Add matrix control/apply paths now.

## 6. Risks

Cached state can be stale and incomplete. The DTO must not claim physical
readback proof or make matrix writes queueable/live.

## 7. Test plan

- Test full cached matrix coverage without `send`.
- Test missing cache coverage without query attempts.
- Test websocket handler round-trip.
- Run focused, broad, smoke, and full pytest validation.

## 8. Where the other agent may disagree

Kimi may want matrix send-state first. That should follow, but a stable matrix
snapshot gives send-state and physical fixture work a clear schema target.
