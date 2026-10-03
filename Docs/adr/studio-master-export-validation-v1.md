# ADR: STUDIO Master Export Validation v1

## Context

`deliver_master()` measured the written WAV and decoded MP3, but its true-peak export gate was hard-coded to `-1.0 dBTP`. A delivery requested with a stricter ceiling such as `-3.0 dBTP` could therefore exceed the requested mastering ceiling and still pass export validation.

## Decision

Add an evidence-only `validate_master_exports()` gate and call it from `deliver_master()` after WAV write and MP3 decode.

The gate validates:
- requested mastering ceiling, with small format-specific tolerances;
- an absolute repository safety ceiling of `-1.0 dBTP`, which tolerance may never weaken;
- standards-measured integrated loudness (`pyloudnorm`) against the requested target;
- frame count and sample rate for both validated representations.

A passing technical gate does not promote an audio baseline and never implies human mastering acceptance.

## Why this won

The fix closes the mismatch between the controller request and the exported artifacts without changing mastering DSP. It reuses the existing post-write measurements instead of trusting only the in-memory candidate.

Codec tolerance is bounded around stricter requested ceilings but is clamped to the repository-wide `-1.0 dBTP` safety ceiling. Thus a request for `-3.0 dBTP` remains meaningfully stricter while a request for `-1.0 dBTP` cannot be relaxed to `-0.65 dBTP`.

## Rejected alternatives

- Keep the hard-coded `-1.0 dBTP` check: does not enforce stricter user/controller ceilings.
- Compare only the in-memory master: misses quantization/codec changes.
- Let codec tolerance exceed `-1.0 dBTP`: conflicts with repository safety rules.
- Treat technical validation as subjective mastering approval: explicitly prohibited.

## Test plan

Focused tests cover:
- known Belye Stai export metrics at a requested `-1.2 dBTP` ceiling;
- stricter `-3.0 dBTP` rejection;
- bounded lossy tolerance;
- global `-1.0 dBTP` safety clamp;
- standards-loudness requirement;
- missing format/length/sample-rate evidence;
- invalid policy inputs;
- `deliver_master()` integration proving the requested ceiling reaches the export gate.

Full repository CI on Python 3.10/3.11/3.12 is required before merge.

## Boundaries

No mastering DSP parameters, Perceptual Critic thresholds, LIVE code, neural audio, paid external services, or audio baseline are changed. Human listening remains required for subjective acceptance.
