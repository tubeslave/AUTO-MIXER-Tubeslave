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
