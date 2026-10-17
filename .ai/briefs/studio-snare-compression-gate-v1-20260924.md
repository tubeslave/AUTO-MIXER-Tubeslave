# STUDIO brief — Grouped Snare Compression Gate v1

## Why now
Kick has a grouped-mic in-place compression gate and the user asked for instrument-specific compression. The next unblocked drum boundary is `SN_T + SN_B`. The original song recipe already uses bottom polarity inversion, bottom expansion, relative mic balance, one group compressor and final level; experiments must preserve those source-specific decisions rather than stack a generic compressor after `SNARE.wav`.

## Bounded task
- reproduce the full 207 s existing `SNARE.wav` exactly from raw `SN_T/SN_B`;
- expose the original group-compressor insert only;
- freeze bottom polarity/EQ/expansion, mic balance and final gain for candidates;
- detect events once from the frozen pre-compression sum and reuse identical windows;
- require body-level consistency improvement while protecting attack/body contrast, quiet-hit audibility and between-hit bleed;
- rerender technical survivors through the full Belye Stai session with the accepted vocal v2;
- never rank a musical winner or auto-promote audio baseline.

## Out of scope
No live/OSC work, no mastering changes, no neural audio, no paid services, no change to the accepted vocal, and no automatic release promotion.
