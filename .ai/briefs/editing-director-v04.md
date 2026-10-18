# Editing Director v0.4: pitch editing

Pitch correction is note/phrase-aware and deliberately partial.

Pipeline:
1. monophonic F0 diagnosis on each vocal independently;
2. reject low-confidence/unvoiced frames and octave glitches;
3. segment stable notes;
4. infer nearest stable chromatic note locally, without forcing a global scale;
5. correct only stable-note median errors >=18 cents;
6. correction strength is 65% and capped at 35 cents;
7. preserve vibrato, slides, consonants and unvoiced material;
8. exact track length must be preserved;
9. render the identical autonomous mix and compare level-matched against v0.3.

This v0.4 policy is conservative by design. A future key-aware mode may use harmonic context, but chromatic local correction is safer than guessing a scale.
