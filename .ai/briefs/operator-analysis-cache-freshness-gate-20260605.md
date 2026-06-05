# Brief: operator-analysis-cache-freshness-gate-20260605

## Context

Product Layer snapshots now expose cache freshness, but operator analysis can
still turn cached fader/gain values into applyable proposals. A stale or
untimestamped current value is not sufficient evidence for supervised apply.

## Task

Gate operator-analysis fader/gain proposals on fresh cached current-value
timestamps. Observations should remain visible, but stale, unknown, missing, or
unsupported cache evidence must make the proposal analysis-only.

## Safety Boundaries

- No live OSC or dLive MIDI/TCP writes.
- No Product Layer or operator-analysis transport queries.
- Do not widen apply authority beyond current supervised fader/gain.
- Freshness is cache evidence only, not physical readback proof.

## Expected Evidence

- Fresh cached fader proposals remain applyable and carry freshness evidence.
- Stale cached fader proposals become `can_apply=false`.
- Queue serialization preserves `cache_freshness` for requested changes.
- Existing supervised pre-read/readback gates remain unchanged.
