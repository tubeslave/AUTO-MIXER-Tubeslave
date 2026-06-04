# Codex Proposal: Bus Main DCA Read DTO

Task id: bus-main-dca-read-dto-20260604
Date: 2026-06-04

## 1. Short thesis

Expose cached bus, main, and DCA console state as a Product Layer read contract
before any bus/group/DCA balance proposals can be considered for promotion.

## 2. What I understood

The project has proposal-only bus/group balance work, but readback remains too
implicit. Existing WING helper methods can actively query the console, so this
step must not call them. It should only normalize existing cached state.

## 3. Proposed solution

Add `build_bus_main_dca_snapshot` with:

- bus rows for fader, mute, pan, width, mono, EQ, dynamics, main send, tags, and
  DCA assignments;
- main rows for fader, mute, pan, and EQ;
- DCA rows for fader, mute, color, icon, monitor, and identity;
- per-section `complete`, `partial`, or `missing` coverage;
- safety flags: `read_only=true`, `live_write=false`,
  `mutates_console_state=false`, and `queries_console=false`;
- websocket handler `get_bus_main_dca_snapshot`.

## 4. Likely files to touch

- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`

## 5. Alternatives considered

- Reuse `groups_buses_snapshot`.
- Call WING `get_bus_settings` and `get_dca_settings`.
- Add bus/main/DCA apply support now.

## 6. Risks

Cached state can be stale or incomplete. The DTO must not claim physical
readback proof and must not make bus/main/DCA control queueable or live.

## 7. Test plan

- Test full cached bus/main/DCA coverage without `send`.
- Test missing cache coverage without query attempts.
- Test websocket handler round-trip.
- Run focused, broad, smoke, and full pytest validation.

## 8. Where the other agent may disagree

Kimi may prefer physical readback fixtures first. This DTO is still the right
next step because physical fixtures need a stable Product Layer schema to
assert against.
