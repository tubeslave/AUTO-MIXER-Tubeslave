# Editing Director v0.6: performance consistency

Goal: remove technical level outliers before compression without flattening musical dynamics.

- Detect event-level distributions per instrument.
- Use robust median/MAD statistics, not absolute target levels.
- Edit only isolated high-confidence outliers.
- Partial clip-gain correction, max +/-2 dB.
- Preserve intentional accents, fills, crescendos and section changes.
- Drum edits must remain coherent across related microphones when an event is changed.
- Bass/guitar corrections use longer windows than drums.
- Click/pop repair is separate and requires an isolated derivative discontinuity.
- Never normalize every note/hit.
- Re-render with the identical mix chain and A/B against v0.5.
