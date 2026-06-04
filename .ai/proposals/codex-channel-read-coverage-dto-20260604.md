# Codex Proposal: Channel Read Coverage DTO

Task id: channel-read-coverage-dto-20260604
Date: 2026-06-04

## 1. Short thesis

Expose channel read completeness as data, not as an implicit assumption inside
`control_state`. This lets the cockpit and mode gates distinguish usable cached
readback from missing or partial console visibility.

## 2. What I understood

`build_channel_inventory` already reads cached channel names, fader/gain/mute,
pan, gate, compressor, routing defaults, and capture meters without scanning or
writing to the mixer. The missing contract is per-section coverage metadata.

## 3. Proposed solution

Add `read_coverage` to every channel row and an aggregate `read_coverage` block
to the inventory snapshot. The DTO should:

- mark every section `complete`, `partial`, or `missing`;
- publish source metadata such as `mixer_state_cache`, `static_routing`, and
  `audio_capture`;
- list missing fields for each section;
- set `read_only=true`, `live_write=false`, and `mutates_console_state=false`;
- keep all existing channel fields unchanged.

## 4. Likely files to touch

- `backend/operator_product_state.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`

## 5. Alternatives considered

- Leave coverage implicit in `None` control fields.
- Add a separate endpoint just for coverage.
- Promote channel read coverage to complete.

## 6. Risks

The DTO could be overread as full channel readback. The inventory must remain
`partial` because color/icon, source patch, HPF, EQ, and stale-cache timestamps
are still not fully normalized.

## 7. Test plan

- Prove cached control and processing sections report `mixer_state_cache`.
- Prove missing cache sections report `missing` and `unavailable`.
- Prove no mixer send or channel-name scan is performed.
- Run focused, broad, smoke, and full pytest validation.

## 8. Where the other agent may disagree

Kimi may ask to implement channel EQ/HPF/source-patch DTOs immediately. That is
valid follow-up work, but this smaller contract is the right next increment
because it labels the existing read surface before widening it.
