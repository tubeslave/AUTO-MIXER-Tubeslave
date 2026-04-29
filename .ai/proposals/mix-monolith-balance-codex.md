# Codex Proposal: Mix Monolith-Style Auto Balance

## Short Thesis

Implement Mix Monolith's public "level-plane" idea as a safe, configurable layer on top of the existing LUFS Auto Balance rather than trying to clone proprietary plugin internals.

## What I Understood

The user wants instrument and vocal fader balancing to behave more like AYAIC Mix Monolith: learn channel levels, place sources forward/backward in a mix plane, and support repeated passes for better balance.

## Proposed Solution

- Normalize instrument labels (`leadVocal`, `electricGuitar`, etc.) into stable level-plane keys.
- Resolve targets as `level_plane_base_lufs + instrument_offset_db`.
- On pass 2+, estimate virtual group/bus loudness and trim multi-channel groups within a small limit.
- Default the fader ceiling to `0 dB` for live-sound safety.
- Preserve the old profile target fallback for unknown instruments and when level-plane mode is disabled.

## Likely Files To Touch

- `backend/auto_fader.py`
- `config/default_config.json`
- `tests/test_auto_fader_level_plane.py`
- `.ai/briefs/mix-monolith-balance.md`
- `Docs/adr/mix-monolith-balance.md`

## Alternatives Considered

- Full plugin emulation: rejected because internals are proprietary and unnecessary.
- Replacing existing Auto Balance: rejected because the current LUFS collection and bleed handling already match project architecture.
- Enabling positive fader boosts above unity: rejected by default due live-sound safety rules.

## Risks

- Existing mixes may feel slightly different on repeated Auto Balance passes because group trim now applies on pass 2+.
- Some custom instrument names may fall back to legacy profile targets until aliases are added.

## Test Plan

- Focused pytest for level-plane targets, group trim, boost limit, and unity ceiling.
- Python compile check for `backend/auto_fader.py`.
- Standard full pytest command before merge.

## Where The Other Agent May Disagree

Kimi may prefer keeping the new behavior disabled by default. I recommend enabling it because the default offsets intentionally mirror the existing profile targets closely, while adding safer fader clipping and second-pass group behavior.
