# ADR: baseline-aware bass compression stability gate v1

## Decision
For bass compression iteration, compare every new first-stage setting against the current known-good first compressor instead of assuming that a generic role preset is an improvement. Detect macro events once from the pre-compression bass and reuse those windows for baseline and candidates.

The proxy gate measures:
1. P90-P10 RMS spread of fixed post-event body windows;
2. median attack/body RMS contrast in the same fixed event set.

A candidate technically survives only if body spread improves by at least 0.10 dB and attack/body contrast does not fall by more than 0.40 dB. These are engineering proxies, not listening scores.

## Candidate family
`no_change` is an explicit reference. The three probes change only attack/release around the existing compressor:
- faster recovery: release ×0.80;
- longer recovery: release ×1.30;
- quicker attack + longer release: attack ×0.80 and release ×1.30.

Threshold, ratio, knee, detector mode, RMS integration and max-GR remain frozen. This isolates timing before attempting another strength change.

## Belye Stai evidence
The previous generic lower-GR family worsened full-song body spread. An initial stronger-GR baseline-aware family also worsened it. A 60 s screen briefly suggested 7 ms / 170 ms, but the full 207 s validation reversed the result, increasing body spread by +0.100 dB. That candidate was rejected.

The final timing-only full-song family also produced no survivor:
- faster recovery: body-spread delta +0.0159 dB, attack/body delta -0.0196 dB;
- longer recovery: +0.00383 dB, +0.0136 dB;
- quicker attack + longer release: +0.0899 dB, -0.0692 dB.

Therefore Belye Stai retains the existing bass compressor. This is a successful fail-closed outcome, not evidence that bass compression is universally solved.

## Guardrails
Full routed rerender remains mandatory before any musical audition. Human listening remains mandatory for subjective acceptance. No candidate can automatically replace the baseline.
