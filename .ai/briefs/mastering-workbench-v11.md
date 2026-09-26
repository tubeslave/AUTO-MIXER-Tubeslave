# Mastering Workbench v1.1: autonomous loudness-budget allocation

Input intent may be as short as:
"loud sections around -8 LUFS, final limiter <= 3 dB GR."

The optimizer searches a bounded budget across:
- density/bus compression,
- multiband peak conditioning,
- oversampled clipping,
- pregain,
- final limiter.

Constraints:
- hit loud-section target within tolerance;
- never exceed final-limiter hard GR budget;
- candidate must pass musical regression;
- prefer the least costly distributed processing, with limiter GR penalized most heavily;
- do not treat crest reduction as damage by itself;
- full-song validation can reject a representative-window winner;
- human level-matched listening remains final acceptance.

The optimizer chooses a processing budget, not a subjective winner.

## Mastering Director v2 safety gate

The base mastering render now records an explicit inter-sample true-peak estimate using bounded 4x polyphase oversampling. The mastering decision layer protects true peak, crest, stereo width and correlation as machine-safety evidence.

A technically safe mastered candidate is not a subjective acceptance. Because mastering is audibly transformative, the default verdict is `pending_human_review`; `baseline_eligible` stays false until a later human-listening transition accepts the candidate. A true-peak ceiling violation or protected regression yields `rejected` and must not advance the baseline.

Validation for the implementation commit set ending at `ace60d19dbb789788ae6ed8b313c617c616bd280`: Stem Offline Test run 35893635585 passed, and full Tests run 35893635712 passed on Python 3.10, 3.11 and 3.12.

## Mastering budget evidence v2

When a LUFS target is configured, the mastering safety gate now requires a standards-based integrated loudness measurement from `pyloudnorm`. Missing loudness evidence is not replaced by an RMS approximation and therefore cannot silently satisfy a LUFS target. The measured target error must remain inside the configured tolerance.

When the maximizer is enabled, its existing per-band and final gain-reduction diagnostics are mandatory safety evidence. The default hard budgets are <= 3.0 dB final limiter GR and <= 4.0 dB worst-band GR. Exceeding either budget rejects the candidate before subjective listening. The render report carries target LUFS, measured LUFS and method, target error, final limiter GR and worst-band limiter GR.

These checks are machine-safety and goal-compliance gates only. A technically safe master still remains `pending_human_review`, and `baseline_eligible` remains false until level-matched human listening accepts it.

Validation for implementation commit `864508aaa8bd23f5069517b7954836d3a696cd9d`: local targeted mastering tests passed 12/12; Stem Offline Test run 35900568898 passed; full Tests run 35900568906 passed on Python 3.10, 3.11 and 3.12.
