# SONG REPA Accepted FX15 Air -0.80 Pipeline

This document records the operator-accepted SONG REPA render as a successful baseline for future offline mixes and for the new decision/correction pipeline.

## Golden Render

- WAV: `/Users/dmitrijvolkov/Desktop/Ai MIXING/SONG_REPA_MIXING_STATION_VIS_FX15_AIR_MINUS_0_80_20260427.wav`
- MP3: `/Users/dmitrijvolkov/Desktop/Ai MIXING/SONG_REPA_MIXING_STATION_VIS_FX15_AIR_MINUS_0_80_20260427.mp3`
- Report: `/Users/dmitrijvolkov/Desktop/Ai MIXING/SONG_REPA_MIXING_STATION_VIS_FX15_AIR_MINUS_0_80_20260427_report.json`
- SHA256: `6d53d5a874560c26200097bc1459d17f052963a9a405c4da13495330abab1f92`
- Metrics: `-14.37 LUFS`, `-4.979 dBFS` sample peak, `-16.951 dBFS` RMS, `204.747 s`, `48 kHz`, stereo.

## How It Was Built

The accepted render is not a raw unity-sum or a generic optimizer pass. It is the result of the prior successful SONG REPA chain:

1. Full project agent/source-knowledge/AutoFOH/phase stack, with `large_system_polish` disabled after MuQ-A1 A/B testing.
2. Drum phase alignment to the overhead pair and snare-bottom polarity support inside the layer-group sum.
3. Event-based expansion on kick, toms, and lead vocal.
4. Cross-adaptive EQ with the lead vocal protected as highest priority and accompaniment receiving anti-mask cuts.
5. Kick/bass hierarchy pass: kick protected as low-end anchor and bass trimmed only around the overlap.
6. Shared filtered FX: vocal plate, short drum room, ducked tempo delay, and chorus doubler.
7. Reference-guided vocal FX focus using `/Users/dmitrijvolkov/Desktop/reference/Theory_Of_A_Deadman-Santa_Monica-spaces.im.mp3`.
8. FX15 section-selective subtle modulation/space on low-MuQ windows `63-102s`, `128-166s`, and `178-202s`.
9. Final master around `-14 LUFS` with no clipping.
10. Small reversible final air trim of `-0.80 dB`.

## Why It Was Accepted

The FX15 final had MuQ-A1 `0.7725038714706898` and technical final score `0.840752705599214`. The strict near-0.78 print, after the final air trim, reached MuQ-A1 `0.778733894` with peak around `-4.979 dBFS`. The provided commercial reference scored `0.7958870023488999` with the same evaluator, so forcing a `0.9` gate on this material was explicitly rejected as unrealistic.

On 2026-04-28, the new AI critic postcheck rendered extra postmaster candidates from this accepted file:

- AYAIC output finish / mirror-EQ
- light glue compressor
- mud/harsh cleanup
- safety trim

The decision layer should select the accepted render as `candidate_000_no_change`, because none of the new candidates improved enough over the accepted baseline. That is now the intended behavior: preserve the golden recipe unless a new candidate clearly beats it and Safety Governor accepts it. Recipe details belong in metadata/report text, not in the no-change candidate id.

Postcheck output:

- `/Users/dmitrijvolkov/Desktop/Ai MIXING/SONG_REPA_FX15_BASELINE_PLUS_AI_CRITICS_20260428_153259/renders/best_mix.wav`
- `/Users/dmitrijvolkov/Desktop/Ai MIXING/SONG_REPA_FX15_BASELINE_PLUS_AI_CRITICS_20260428_153259/reports/summary_report.md`

## Mixing Station Note

The Mixing Station visualization report sent only verified WING Rack `fader` and `pan` paths. HPF, EQ, sends, compressor, and main-air paths were logged and blocked as `needs_discovery`. Do not turn those blocked live-console entries into OSC writes until the exact WING dataPaths are discovered and read back.

## Rule For Future Mixes

For SONG REPA, start from this recipe or from the exact golden render as `candidate_000_no_change`. For similar rock multitracks, reproduce the same order first:

1. AYAIC-style input level-plane balance when building from raw stems.
2. Agent/source-knowledge/AutoFOH/phase stack.
3. Kick/bass hierarchy.
4. Cross-adaptive EQ with vocal protection.
5. Reference-guided vocal FX and filtered shared returns.
6. Section-selective modulation/space candidates.
7. New decision-layer candidate corrections.
8. Critic evaluation and Safety Governor acceptance.

Do not replace this with a raw unity-sum, broad postmaster EQ, heavy limiting, or loudness tricks. New pipelines may add candidates, but the golden candidate stays in the comparison set and can win as `no_change`.
