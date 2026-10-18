# ADR: Belye Stai Tom Compression Gate v1

## Decision
Treat `TOM_1`, `TOM_2` and `FLOOR` as three independent close-mic sources with the delivered preparation frozen before their compressor. Candidate processing replaces the existing compressor in place. It does not borrow vocal settings and does not add a second compressor after the processed tom.

## Frozen source path
Each source keeps its delivered high-pass (`65 / 55 / 45 Hz`), 10.5 kHz low-pass, `420 Hz -2 dB Q0.8` bell, 80–350 Hz expander and downstream pan (`-0.45 / +0.12 / +0.52`). The original compressor baseline is `2.6:1`, `18 ms` attack, `160 ms` release, `5 dB` knee and `4 dB` max GR, with its source-specific threshold and final gain reproduced from the baseline render.

## Technical gate
Events are detected once from the prepared pre-compression source and identical windows are reused for baseline and candidates. The gate measures body-level spread, attack/body contrast, body/tail decay shape, quiet-hit level and between-hit floor. Passing a proxy gate means only that the variant is safe enough for a full routed rerender, never that it sounds better.

The first family changes attack/release only: `preserve_attack` = `22.5 / 136 ms`, `tighter_body` = `14.4 / 128 ms`, `longer_decay` = `18 / 200 ms`. Threshold, ratio, knee, max GR, detector and final static gain stay frozen.

## Real-song evidence
All three no-change adapters reproduce the previous processed sources sample-for-sample. On the full 207 s song, `preserve_attack` is the only technical survivor for all three close mics. Body-spread changes are approximately `-0.206 / -0.244 / -0.246 dB` for TOM_1 / TOM_2 / FLOOR. Attack/body changes remain within about `0.023 dB`, body/tail changes within `0.049 dB`, and between-hit floor increases are `0.223 / 0.129 / 0.135 dB`, below the predeclared `0.40 dB` guard.

A full routed premaster with the previously human-preferred vocal insert and all three `preserve_attack` tom candidates has no protected Perceptual Critic failures. Whole-mix deltas are tiny: foreground `-0.00022 dB`, vocal-intelligibility proxy `+0.000068`, punch proxy `-0.0385 dB`, harshness `-0.00030`, depth proxy `+0.00105`, width `-0.00316 dB`, density and climax lift unchanged.

## Boundaries
No automatic musical winner or audio-baseline promotion. Human level-matched listening is mandatory. Existing vocal, bass, kick, snare, mastering and live paths are not modified by this task. No neural audio or paid external credits.