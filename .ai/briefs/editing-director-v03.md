# Editing Director v0.3: vocal cleanup

After phase/polarity and groove editing, vocal editing remains minimal-intervention.

1. Detect vocal phrases from vocal-band energy with hysteresis.
2. Between phrases, attenuate stage bleed/noise softly; never hard-gate.
3. Preserve breaths and reverb tails with slow soft edges.
4. Correct only phrase-level level outliers, max +/-2 dB. This is editing, not compression.
5. Do not align two vocal tracks unless analysis supports that they are actual doubles of the same phrase.
6. Pitch remains diagnose-only until artifact-safe correction is validated.
7. Preserve exact track length and re-run the identical autonomous mix for A/B.
