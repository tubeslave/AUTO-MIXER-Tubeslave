# Brief: channel-processing-field-freshness-20260605

## Context

Full autonomous EQ, compression, and gate decisions require the Automixer to
know not just whether cached channel-processing fields exist, but whether each
field is fresh enough to trust. The current channel processing snapshot exposes
coverage and a broad cache freshness summary, but section-level decisions still
cannot distinguish a fresh compressor threshold from stale EQ or missing gate
timing.

## Task

Add read-only per-field cache freshness metadata to the channel processing
snapshot for HPF, EQ, compressor, and gate sections. Keep the surface
proposal-only/read-only and update the inventory blocker so future work knows
that physical query fixtures and WING/dLive parity remain open.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- No new proposal queue import path or live executor for EQ, compressor, gate,
  HPF, or de-esser.
- Do not query the console from product-state snapshot builders.
- Preserve the current supervised write boundary: only `fader` and `gain`
  through the manual gate.

## Expected Evidence

- HPF/EQ/compressor/gate read coverage carries field-level freshness status.
- Stale processing cache timestamps are visible at section and field level.
- Existing no-query/no-send tests continue to prove the DTO is read-only.
- Read surface inventory still blocks full autonomous readiness.
