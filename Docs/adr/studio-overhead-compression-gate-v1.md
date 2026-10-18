# ADR: OH / Cymbal Compression Gate v1

## Decision
Overheads use a no-change-first policy. The delivered Belye Stai OH recipe contains no compressor. A linked stereo compressor may be proposed only when fixed high-band event evidence demonstrates an actionable inconsistency in bright cymbal peaks.

The song adapter reproduces the existing OH path before any candidate: `OHL` plus `OH_R` calibrated +3 dB, 190–13.5 kHz filtering, a -1.2 dB bell at 6 kHz (Q 0.7), and the original active-level stage. Candidate compression is inserted before the output stage and the baseline final gain from the same exact source pair is frozen, so compression cannot be hidden by re-leveling.

## Evidence and gates
Detection is fixed from the reference stereo source in approximately the 3.2–12 kHz band and reused for baseline/candidate measurement. Candidate assessment protects:

- bright peak-excess P90–P10 consistency;
- the P95 upper tail of bright peak excess;
- 90–260 ms cymbal decay shape;
- pre-event floor relative to event body, to catch ambience/bleed pumping;
- 2 s macro dynamics;
- integrated side/mid energy and L/R correlation.

Only attack/release may change in the initial family. Threshold, ratio, knee, maximum GR, detector mode, RMS window and sidechain HPF remain frozen. No makeup gain, clipping, width processing or automatic level matching is part of candidate rendering.

## Validation
A 24 s synthetic linked-stereo exercise validates the control logic, not musical quality. The stable case produced 49 fixed events and 12 active blocks; bright peak-excess spread was 1.5467 dB, so the gate returned `no_change`. The deliberately unstable case produced 60 events and 12 active blocks; spread was 24.2867 dB and the problem was actionable.

All three bounded candidates improved bright-peak consistency, by about 0.27 / 0.49 / 0.79 dB respectively, but all were rejected by the predeclared inter-event floor/body guard. That relationship changed by about 0.78 / 0.85 / 0.92 dB, above the 0.25 dB limit. No threshold was loosened after seeing the results. A technical failure here is a successful fail-closed decision.

## Real-song limit
The preserved Belye Stai source manifest identifies the exact source pair (`11_OHL.wav`, SHA-256 `8db4b38fbff7f3c8e1471911be72a283a8a18ca3a3fa56805bd77ef25a6fdf16`; `10_OH_R.wav`, SHA-256 `994c93094b97fb1458cc99ed19fdab602913f1981449294d86d386a4ec314126`; 9,128,700 frames at 44.1 kHz). Their raw WAV bytes are not present in the current execution runtime. Therefore this task does not claim a real-song OH actionability score, a new Belye Stai OH render, a full-session Perceptual Critic result, or a listening A/B.

## Safety boundaries
- Explicit stereo input is mandatory and compressor gain reduction is linked across L/R.
- Fixed detection events are reused for all comparisons.
- No-change is a valid successful outcome.
- A technical survivor must still pass the full routed session and protected Perceptual Critic checks.
- Human listening remains mandatory for subjective acceptance.
- No automatic audio-baseline promotion.
- Accepted vocal processing, mastering, live DSP and shared critic thresholds are outside this task.
- No neural audio processing or paid external services are required.
