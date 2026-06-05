# Brief: recommendation-bridge-cache-freshness-gate-20260605

## Context

SafeGain and soundcheck recommendations can be imported into the operator
proposal queue as fader/gain actions. These recommendations are central to
autonomous soundcheck and gain staging, but their current console values must be
fresh when runtime mixer cache is available.

## Task

Add a cache freshness promotion gate to recommendation bridge proposals. Fresh
runtime cache should preserve applyable fader/gain changes and add evidence;
stale, unknown, or missing runtime cache should keep the recommendation visible
but make it analysis-only.

## Safety Boundaries

- No live OSC or dLive MIDI/TCP writes.
- No bridge-level mixer queries.
- Do not widen apply authority beyond current supervised fader/gain.
- Pure adapter tests without a runtime `state` cache remain conversion-only.

## Expected Evidence

- SafeGain fresh trim proposals carry `cache_freshness`.
- SafeGain stale trim proposals become analysis-only.
- Soundcheck import path passes the runtime mixer client into the bridge.
- Soundcheck stale fader recommendations are blocked from applyability.
