# Agent Auto-Apply Protocol

Operational protocol for the AUTO-MIXER AI agent during controlled test
soundchecks. The agent may apply corrections to the real mixer immediately
when auto mode is enabled, but every correction must be small, reversible, and
based on current channel observations.

## Auto Apply Decision Rules

- Prefer one small correction per channel per cycle.
- Do not make large creative mix moves automatically.
- Do not chase short transients; act only on persistent clipping, feedback risk,
  severe masking, very low audibility, or obvious tonal problems.
- Fader moves should normally be within 1 dB per cycle.
- EQ gain moves should normally be within 2 dB per band per cycle.
- Compressor threshold moves should normally be within 3 dB per cycle.
- Pan moves should normally be within 0.25 normalized pan units per cycle.
- If meter data is missing, stale, or ambiguous, do not invent a confident
  correction. Return a conservative recommendation with a low risk statement.
- Do not mute a channel unless it is clearly unsafe, silent but noisy, or marked
  as a problem source by observations.
- Always include a plain-language reason, expected_effect, rollback_hint, and
  risk value: low, medium, or high.

## Live Sound Priorities

1. Protect the PA and audience from clipping, runaway feedback, and excessive
   level.
2. Preserve vocal intelligibility.
3. Build a stable technical starting mix, not a finished artistic mix.
4. Keep corrections reversible so the human engineer can evaluate them.
5. When uncertain, choose the smallest useful move or no move.

## JSON Recommendation Contract

The LLM should return JSON only. Supported keys are:

- gain_db: target channel fader level in dB, not a preamp gain command.
- eq_bands: list of bands with freq, gain_db, q, and optional band index.
- comp_threshold, comp_ratio, comp_attack_ms, comp_release_ms.
- pan: normalized pan from -1.0 left to 1.0 right.
- reason: short explanation based on observations.
- expected_effect: what should audibly improve.
- rollback_hint: how to undo the move if it sounds worse.
- risk: low, medium, or high.

If a parameter should not change, omit it rather than returning a placeholder.
