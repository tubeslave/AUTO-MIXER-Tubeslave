# Codex Proposal: Routing Sends Read DTO

Task id: routing-sends-read-dto-20260604
Date: 2026-06-04

## 1. Short thesis

Expose route and send read coverage before any routing/send automation can be
promoted. Static Dante intent and cached WING state are useful evidence, but
not live authority.

## 2. What I understood

The repo has static Dante role mapping and WING helpers for sends and output
routing. Some routing helpers send OSC queries or perform writes, so Product
Layer must not call them in this step.

## 3. Proposed solution

Add `build_routing_sends_snapshot` with:

- static Dante route rows from `get_routing_as_dict`;
- cached channel bus sends from `/ch/*/send/*`;
- cached channel main sends from `/ch/*/main/*`;
- cached output route sources from `/io/out/*`;
- destructive-change markers on output route rows;
- safety flags: `read_only=true`, `live_write=false`,
  `mutates_console_state=false`, and `queries_console=false`;
- websocket handler `get_routing_sends_snapshot`.

## 4. Likely files to touch

- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`

## 5. Alternatives considered

- Leave routing as static Dante prose only.
- Call `get_output_routing`.
- Make routing/send proposals queueable now.

## 6. Risks

Cached route/send state can be stale or partial, and routing writes can repatch
critical outputs. The snapshot must remain read-only and mark destructive
change implications without adding execution paths.

## 7. Test plan

- Test static and cached route/send rows without `send`.
- Test missing cache behavior without query attempts.
- Test websocket handler round-trip.
- Run focused, broad, smoke, and full pytest validation.

## 8. Where the other agent may disagree

Kimi may ask for diff artifacts before DTO work. The DTO is still a necessary
schema target for route/send diff artifacts and physical-query fixtures.
