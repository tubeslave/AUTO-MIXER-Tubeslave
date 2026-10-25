# ADR: KEYS / PLAYBACK Compression Gate v1

## Decision
KEYS and PLAYBACK are separate dynamics roles. Neither inherits vocal, bass, drum or guitar compressor settings.

KEYS compression is considered only after a local dynamics problem is demonstrated and must preserve attack/body contrast plus stereo side/mid and L/R correlation. PLAYBACK is protected more strongly because its dynamics may already be programmed; large section-to-section changes block compressor proposals and can instead request review of a bounded section-level ride.

No-change is a successful outcome. Candidate generation changes only attack/release relative to the supplied baseline compressor; threshold, ratio, knee, detector and max-GR remain frozen. A technical survivor still requires complete routed-session rendering and human listening.

## Evidence
Synthetic stereo validation exercises three cases: intentionally uneven transient KEYS, stable PLAYBACK, and PLAYBACK with deliberately programmed macro section changes. The KEYS case demonstrates that the gate can admit bounded candidates and measure stability, transient and stereo guards. Stable PLAYBACK produces no candidates. Programmed macro PLAYBACK blocks compression and returns `review_section_level_ride_not_compression`.

This evidence validates control logic only. The current execution runtime does not contain the original Belye Stai `KEYS_L/KEYS_R/PLAYBACK_L/PLAYBACK_R` WAVs, so no song-level result or musical preference is claimed.

## Safety boundaries
- Explicit stereo input is mandatory.
- Detection windows/events are fixed from the reference source and reused for candidate comparison.
- Thresholds are not weakened after candidate results are observed.
- Stereo image, macro dynamics and transient contrast are protected.
- No automatic baseline promotion.
- Human listening remains mandatory for subjective acceptance.
- No neural audio processing or paid external services are required.
