# Brief: P1 — fader ceiling defence-in-depth at WingClient level

**Author:** Kimi
**Date:** 2026-07-19
**Priority:** P1 (review roadmap item 11)
**Depends on:** P0 + C3 (merged)

## Context

Review finding (High #7): `WingClient.set_channel_fader` (`backend/wing_client.py:720`) clamps only the hardware range [-144, +10] dB. The 0 dBFS ceiling mandated by CLAUDE.md ("never set fader > 0 dBFS without explicit operator request") is enforced only by upper layers (auto_fader, heuristics, manual-write config). Any raw programmatic caller — a bug, an unclamped agent, a tool — can write +10 dB to a live console channel. Defence-in-depth requires the ceiling at the transport boundary itself.

## Design constraint discovered during planning

A per-call `allow_above_unity` flag would require changing the `MixerClientBase` contract, `ObservationMixerClient`, four duck-typed test doubles (including `test_manual_write_wal.py`, which is in Codex's active M-thread scope), and five voice-path call sites — wide blast radius and merge-conflict risk with the parallel Codex thread. Chosen instead: **client-level ceiling** (`fader_ceiling_db` constructor param + `AUTOMIXER_FADER_CEILING_DB` env, default 0.0) — zero signature-contract breakage, one production file touched.

## Scope (Kimi implements)

- `WingClient.__init__`: keyword-only `fader_ceiling_db: Optional[float] = None`; resolution order: param → `AUTOMIXER_FADER_CEILING_DB` env → `0.0`. Hardware clamp [-144, +10] unchanged and applies after the ceiling.
- `set_channel_fader`: values above the ceiling are clamped with a WARNING log (visible, not silent).
- Tests in `tests/test_wing_client.py` (mocked OSC, existing style).
- CLAUDE.md fixed-rules entry; ADR; focused commits.

## Non-goals / behavior notes

- Operator manual-write and voice paths keep the documented `min(10.0, …)` request clamp, but the console write now lands at the client ceiling (default 0 dBFS) with a warning unless the operator explicitly raises `AUTOMIXER_FADER_CEILING_DB`. This is the intended policy hardening, documented in the ADR.
- Other mixer clients (Ableton, dLive) untouched — the safety canon is WING-specific.
