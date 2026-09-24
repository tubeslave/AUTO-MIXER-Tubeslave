# ADR — Belye Stai grouped snare compression gate v1

Status: accepted for machine validation, musical candidate requires human listening.

## Decision
Treat `SN_T` and `SN_B` as one synchronized source group. The adapter reproduces the delivered preparation exactly: bottom polarity inversion, source-specific filtering/EQ, bottom expander, source-bound relative bottom gain, group compressor, then source-bound final gain. Candidate compression replaces only the original group compressor.

The gate detects snare events once from the frozen pre-compression group sum and measures the same windows in baseline/candidates. A candidate must improve P90–P10 body-level spread by at least 0.08 dB while not reducing median attack/body contrast by more than 0.35 dB, lower-quartile attack level by more than 0.40 dB, or increasing the between-hit floor by more than 0.40 dB. These are engineering guardrails, not preference scores.

## Real-song evidence
On the full 207 s Belye Stai source, no-change local processing is sample-identical to the existing processed SNARE. The bounded timing family produced one technical survivor: `tighter_body` (11.2 ms attack / 88 ms release vs 14 / 110 ms baseline). It improved measured body spread by 0.2630 dB, changed median attack/body by +0.0356 dB, quiet-hit proxy by +0.0685 dB, and between-hit bleed proxy by +0.0923 dB. Two other variants failed the body-stability target.

The survivor was rerendered through the full session with the accepted vocal v2. Existing Perceptual Critic protected metrics showed no failure. Therefore it is eligible only for level-matched human A/B. No audio baseline is promoted automatically.

## Constraints
Ghost-note protection is a lower-attack-quartile proxy, not semantic ghost-note transcription. Between-hit floor is a bleed proxy, not source separation. Human listening remains mandatory.
