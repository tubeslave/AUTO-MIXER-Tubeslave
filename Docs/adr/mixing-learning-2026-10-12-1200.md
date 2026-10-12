# ADR: constrained-playback evidence must not become a universal bass rule

- date: `2026-10-12`
- update: `ML-2026-10-12-1200`
- status: accepted for knowledge-base governance; no runtime adoption

## Decision

Treat the reviewed virtual-bass method as evidence only for explicitly bandwidth-limited playback. A harmonic/distorted bass return may be tested as either a creative full-range layer or a constrained-playback translation layer, but the intent must be declared before rendering and the two claims must not be merged.

Any project A/B must retain raw measurement renders and create separate BS.1770-matched blind listening copies. A constrained-playback improvement cannot be promoted as a general mix rule if full-range playback regresses.

The public MATLAB artifact is not eligible for integration: no repository-level licence was found, exact paper parity is unconfirmed, dependencies are unpinned, and the author reports noncausal approximately 150 ms delay plus impractical real-time cost for the tonal path.

## Consequences

- `RULE-ML-20261012-0900-01` remains a bounded shadow candidate with `auto_apply:false`.
- `EXP-ML-20261012-0900-01` remains `not_run_blocked` and gains separate full-range and 150 Hz high-pass stress strata.
- The paper's numeric settings remain source-specific test conditions, not presets.
- No commercial example audio is copied, measured or added as a project reference.
- No audio, DSP, runtime or model change is authorized.
