# STUDIO Compression Director v1

## Context
Compression DSP v2 is machine-validated, but its automatic role threshold and time constants remain engineering defaults. The accepted state explicitly names the next goal: use event/body/recovery evidence to propose bounded attack/release/threshold candidates and make level-match/headroom conflicts explicit.

## Bounded task
Add an opt-in deterministic Compression Director. It may measure macro envelope timing and propose a small audition set, but it must not rank a musical winner, auto-promote a baseline, alter live/OSC/backend processing, or require learned models or paid services.

The director must:
- be phase-safe for linked stereo evidence;
- aggregate macro events instead of waveform-scale ripples;
- propose only bounded `CompressorConfig` candidates;
- keep every audible candidate behind human listening review;
- separate compressor-only rendering from ride/makeup;
- report exact RAW-active RMS match gain separately from any export true-peak constraint;
- support fair A/B by exact level match followed by a common pair trim, not by limiting one side differently.

## Acceptance evidence
Synthetic tests cover onset sensitivity, event-density/release sensitivity, polarity invariance, sustained/silent fallbacks, config bounds, compressor rendering, level-match ceiling conflicts, common-pair headroom trim and quality-loop human gate. Real-source evidence must use unchanged existing Belye Stai WAVs and is audition evidence only, not a new mix baseline.
