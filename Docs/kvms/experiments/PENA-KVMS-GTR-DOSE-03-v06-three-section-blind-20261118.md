# PENA-KVMS-GTR-DOSE-03 — three-section guitar-balance experiment (2026-11-18)

**Status:** render and engineering validation complete; **human listening not yet performed**. Research branch only. `auto_apply:false`; production DSP, console hardware, OSC and main branch unchanged.

## Motivation and prior observations

- v0.3: same listener preferred Y control without vocal-conditional guitar band EQ; feedback “Гитар слишком много”.
- v0.4: preferred X with combined attenuation of Crunch −3.0 dB, Rhythm −3.0 dB, Lead −2.0 dB, Lead2 −4.0 dB.
- v0.5: preferred X vs v0.4, with a further −0.8 dB on Crunch and Rhythm. Two pairwise preferences are by **one listener on the same excerpt**.

## Testable refinement H7b: contextual nonlinear guitar-balance preference

Further reductions may cease to be preferred or reduce desired musical drive. Preference may differ between contrasting sections. Do **not** infer a monotonic law from the first two selections.

v0.6 candidate: further **−0.8 dB Crunch and Rhythm vs v0.5**, all other role gains and DSP fixed. Relative to original Y control: Crunch/Rhythm −4.6 dB, Lead −2.0 dB, Lead2 −4.0 dB.

## Verified offline results

| Mix | Integrated LUFS | True peak dBTP | LRA LU | Guitar-group / vocal RMS on 51.28–77.28 s |
|---|---:|---:|---:|---:|
| v0.4 | −22.6 | −5.6 | 7.6 | +3.45 dB |
| v0.5 (preferred baseline) | −22.8 | −5.7 | 7.5 | +2.94 dB |
| v0.6 (new, **not preferred yet**) | −23.0 | −5.7 | 7.4 | +2.47 dB |

Input: 48 FLAC tracks from edited `Logoped_Pena_Cut_48_Tracks.zip`; all 48 SHA-256 hashes matched the original manifest, 42 non-empty files selected as in previous versions. Export: stereo PCM24 48kHz, 195.81 s, no mastering or final limiter. Sample clipping/NaN: none. SHA-256 v0.6 full WAV: `3838d60e3d4b0ba03465bc21456e8d637af72c40a8e54ca9d690ed1e1cec96a4`.

The source-code diff between v0.5 and v0.6 changes only musical group gains (both −3.8 → −4.6), output paths and explanatory metadata.

## Three sections for blind comparison

Each section includes **three** loudness-matched variants (v0.4/v0.5/v0.6); the letter mapping is independently counterbalanced in each section, and the key remains private until listener feedback.

- S1: 26–46 s, early vocal passage with less guitar density.
- S2: 52–72 s, dense guitars with vocal.
- S3: 172–192 s, final passage with strong guitar activity.

Nine PCM24 clips are normalized to −20.000 LUFS, within-section differences <0.01 LU, plus nine MP3 320k convenience copies. No gain limiter in the test. The private blind key is deliberately NOT shipped in the listener ZIP or recorded in this GitHub note, to preserve blinding.

Full version v0.6 and all blind materials: `Pena_KVMS_v06_H7_Research_Package.zip` (local user handoff; no copyrighted audio committed to repository). SHA-256 ZIP: `23c9a26dc700f31947dccacd5c988d360c7a0d9c97407cc55703781da7502e68`.

## Decision rule

The **incumbent preferred full mix remains v0.5**. Do not mark v0.6 better merely because a guitar/vocal ratio fell. Obtain independent listening choices per section, accept ties, reveal keys only after voting, update H7 with section-dependent observations. This is a single-listener exploratory experiment; not a calibrated objective quality/reward function, not an independently validated mixing rule.

Next work after feedback: test the existence of a plateau/reversal, then replicate across other songs/listeners before changing Automixer decisions.