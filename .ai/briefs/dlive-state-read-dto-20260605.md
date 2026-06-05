# Brief: dlive-state-read-dto-20260605

## Context

The project target includes both Behringer WING and Allen & Heath dLive, but the
Product Layer currently exposes richer WING-oriented read snapshots than dLive
MIDI/TCP state. `DLiveClient` keeps a small local cache for names, faders, mutes,
and raw MIDI events.

## Task

Expose a Product Layer dLive snapshot using cached MIDI/TCP state only. The DTO
must not request channel names, call `get_fader`, send MIDI, recall scenes, or
widen live write authority.

## Safety Boundaries

- No live dLive MIDI/TCP writes.
- Do not call transport helpers such as `request_channel_names`, `send`,
  `_send_raw`, or any setter from the Product Layer builder.
- Keep dLive parity marked partial until EQ, dynamics, routing, scenes, FX,
  physical fixtures, and stale-cache timestamps are normalized.

## Expected Evidence

- Focused tests prove cached dLive state exposure and empty-cache behavior.
- Handler round trip returns read-only safety flags.
- Read surface inventory marks dLive as `partial` Product snapshot coverage.
