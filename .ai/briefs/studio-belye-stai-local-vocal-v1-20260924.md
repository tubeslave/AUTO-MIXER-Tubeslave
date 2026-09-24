# STUDIO task: Belye Stai Local Vocal Processor Adapter v1

## Why now
Routed Compression Family v1 proved that adding Compression Director candidates after the already-processed `VALERA_VOX` track is the wrong intervention boundary. The next unblocked step is to expose the original raw-to-processed local vocal chain and replace its first compressor in place.

## Bounded task
Extract the frozen `VALERA_VOX` local DSP chain from the delivered procedural `mix_dsp.py` into a reusable STUDIO module. Preserve the no-change path exactly. Expose the signal immediately before the first compressor and allow one explicit `CompressorConfig` replacement at that point. Keep the original second compressor, de-esser and final level stage unchanged.

## Acceptance evidence
- Real 207 s raw vocal reproduces the procedural pre-compression signal and final processed vocal sample-for-sample.
- Raw input remains immutable.
- Replacement path is explicitly human-review-only and never baseline eligible.
- Focused contract tests pass; exact-branch CI must pass before merge.

No live/OSC work, learned audio inference, paid services, pitch/time correction, mastering changes or musical baseline promotion.