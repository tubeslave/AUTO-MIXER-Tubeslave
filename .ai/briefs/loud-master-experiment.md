# Loud Master experiment

Goal: loud sections near -8 LUFS while final limiter GR never exceeds 3 dB.

Do not ask the limiter to create the entire loudness increase. Distribute crest reduction:
1. gentle density/bus compression;
2. band-local peak conditioning;
3. oversampled soft clipping;
4. final limiter <= 3 dB GR;
5. true-peak ceiling and musical regression checks.

The target is a loud-section target, not necessarily -8 LUFS integrated for the whole song.
Peak reduction can make drums subjectively denser and is not automatically considered damage.
Reject only when the staged process creates measurable/artifact-level transient, spectral,
stereo or true-peak regressions beyond the configured guards.
