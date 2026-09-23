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
