# ADR: Belye Stai grouped-kick compression gate v1

## Decision

Treat KICK_IN and KICK_OUT as one synchronized source group. Reproduce the delivered phase/EQ/mic-balance/local-level recipe exactly, expose the original compressor boundary, and evaluate replacement compression only against the existing grouped-kick baseline.

The gate detects events once from the frozen pre-compression sum and reuses identical windows for baseline and candidates. A candidate must improve body-level spread by at least 0.08 dB while preserving attack/body contrast, quiet-hit audibility and the between-hit floor. Threshold, ratio, knee, RMS detector and maximum GR remain frozen for v1 proposals; only attack/release move within bounded baseline-relative ranges.

## Real-song evidence

On the full 207 s Belye Stai sources, the no-change adapter reproduced the delivered processed KICK sample-for-sample. Baseline compressor threshold was -34.541616 dBFS, ratio 3:1, attack 22 ms, release 115 ms, knee 5 dB, cap 4.5 dB.

Three bounded timing probes were evaluated. `more_punch` and `longer_body` failed the required body-stability improvement. `tighter_body` survived the local technical gate with body-spread delta -0.285833 dB, attack/body delta +0.021536 dB, quiet-hit delta +0.175360 dB and between-hit-floor delta +0.133799 dB.

The surviving kick was then rerendered through the complete Belye Stai session while preserving the human-preferred v2 vocal. The no-change full session re-export matched the delivered v2 premaster byte-for-byte, SHA-256 `73e407ac2893b94ea49aecfa4274da52c3131cc0dafb04fc394c0f94a596eb59`. Full-mix protected deltas for `tighter_body` were small: foreground +0.000339 dB, vocal-intelligibility proxy -0.000217, punch proxy -0.098026 dB, harshness +0.000120, density 0, depth -0.000037 and width +0.009666 dB. No existing protected regression limit was crossed.

## Safety boundary

This evidence means only that `tighter_body` is technically safe enough to audition. It is not a musical winner and cannot promote the audio baseline automatically. Human listening is mandatory. No neural audio or paid external service is involved.
