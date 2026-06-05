# Proposal: dlive-state-read-dto-20260605

## Short thesis

Add a cached-only Product Layer DTO for dLive MIDI/TCP state and keep dLive
console apply blocked.

## What I understood

The autonomous goal requires dLive parity, but current safe live authority is
limited to supervised fader/gain paths. dLive's current client cache is much
narrower than WING OSC state, so the safe step is explicit read coverage, not a
parity claim.

## Proposed solution

Implement `build_dlive_state_snapshot(server)` in
`backend/operator_product_state.py`. It reads `mixer_client.state` and
`_channel_names`, normalizes cached channel names, raw 14-bit fader values,
converted fader dB, mute state, raw note velocity, and raw MIDI cache counts.
Register `get_dlive_state_snapshot` in product-state handlers and update the
read surface inventory to Product snapshot coverage.

## Likely files to touch

- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`
- `tests/test_read_surface_inventory.py`

## Alternatives considered

- Call `get_fader()` and `get_channel_settings()` from the builder. Rejected
  because the Product Layer should expose cache state without transport-shaped
  reads or compatibility methods.
- Request channel names while building the DTO. Rejected because it sends MIDI
  and would make a read DTO mutate network state.
- Claim dLive/WING parity. Rejected because EQ, dynamics, routing, scenes, FX,
  stale-cache timestamps, and physical fixtures remain incomplete.

## Risks

- dLive preamp gain pitchbend cache is transport-channel scoped, not safe
  per-input-channel proof.
- Cached state can be stale until timestamp/readback fixtures exist.
- The DTO exposes only a narrow slice of dLive compared with WING read coverage.

## Test plan

- Unit test populated cached dLive state without transport calls.
- Unit test missing cache behavior.
- Handler round-trip and display-only contract tests.
- Full pytest suite and diff check.

## Where the other agent may disagree

Kimi may prefer a full dLive parity matrix before any DTO. My counterpoint is
that this DTO names the gaps explicitly and gives the cockpit a safe contract to
measure future parity work against.
