# Mix Monolith-Style Auto Balance

## Task

Add output fader balance rules for instruments and vocals inspired by AYAIC Mix Monolith.

## Research Summary

- Official AYAIC material describes Mix Monolith as an automatic mixing system that learns tracks and places them on "level-planes".
- The plugin supports all-channel and per-channel learn/mix workflows, applied gain locking, duck/expand/mute groups, and a two-pass workflow from unity faders to balanced tracks and busses.
- Local machine has `Mix Monolith.component` installed as an AU plugin (`com.Ayaic.MixMonolith`, version `0.6.2`). Only bundle metadata was inspected; no binary reverse engineering was performed.

## Implementation Scope

- Keep the existing Integrated LUFS Auto Balance workflow.
- Add a configurable level-plane target layer: base LUFS plane plus per-instrument offsets.
- Add a second-pass virtual group trim to approximate bus-group balancing.
- Keep live safety: faders default to a `0 dB` ceiling unless `allow_fader_above_unity` is explicitly enabled.

## Test Plan

- Unit tests for instrument normalization and front/back level-plane targets.
- Unit tests for second-pass group trimming.
- Unit tests for fader ceiling and max boost behavior.
- Run focused tests and the standard project test command.
