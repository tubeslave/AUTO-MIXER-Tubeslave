# Mix Monolith-Style Level-Plane Auto Balance

## Context

AYAIC Mix Monolith is publicly described as an automatic mixing system that learns tracks, places them on "level-planes", supports channel/all-channel learn and mix actions, locks applied gain, and uses a two-pass workflow for tracks and busses. The local system controls live-sound faders, so any interpretation must preserve headroom and avoid unsafe gain increases.

## Options Considered

1. Clone the plugin behavior directly.
2. Keep only the existing static target LUFS table.
3. Add a Mix Monolith-inspired level-plane abstraction on top of existing LUFS Auto Balance.

## Decision

Use option 3. Auto Balance now resolves targets through a configurable base level plane plus per-instrument offsets. On pass 2 and later, it estimates virtual group loudness and applies a bounded trim to multi-channel groups.

## Why This Won

It captures the useful public principle without depending on proprietary internals. It reuses existing Integrated LUFS, bleed compensation, channel locks, and pass counting. It also improves live safety by making `0 dB` the default fader ceiling.

## Rejected Alternatives

- Direct plugin analysis or binary reverse engineering: rejected for safety and licensing reasons.
- Replacing Auto Balance wholesale: rejected because it would discard working bleed and LUFS logic.
- Allowing `+10 dB` auto fader moves by default: rejected because project rules require explicit operator intent before going above unity.

## Implementation Plan

- Add instrument normalization and level-plane constants to `backend/auto_fader.py`.
- Add controller settings for level-plane enablement, base LUFS, offsets, two-pass group trim, max boost, and fader ceiling.
- Route `_compute_auto_balance()` through level-plane targets.
- Clip `_compute_auto_balance()` and `apply_auto_balance()` to the configured safe fader range.
- Add focused tests for targets, group trim, boost limit, and unity ceiling.

## Test Plan

- `PYTHONPATH=backend python -m pytest tests/test_auto_fader_level_plane.py -q`
- `PYTHONPATH=backend python -m py_compile backend/auto_fader.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Pass 2+ can lower dense groups more than the previous algorithm. Roll back by setting `level_plane_two_pass_enabled` to `false` or `level_plane_balance_enabled` to `false` in `automation.auto_fader`. The fader ceiling can only exceed unity when `allow_fader_above_unity` is explicitly enabled.
