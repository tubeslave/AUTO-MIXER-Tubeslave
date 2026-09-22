# Editing Director v0.5: harmonic-context pitch editing

Extend v0.4 pitch diagnosis with local harmonic context from playback, guitar, keys and bass.

Rules:
- derive a local chroma profile from accompaniment only, never from the vocal being corrected;
- use harmony to validate/disambiguate nearby note targets, not to force a global scale;
- preserve chromatic/non-diatonic notes when locally supported;
- require stronger evidence for corrections above 20 cents;
- keep partial correction and the 35-cent hard ceiling;
- preserve vibrato, slides, consonants and exact track length;
- do not manufacture a correction when accompaniment is harmonically ambiguous;
- render from the validated v0.4 edit state and compare against v0.4 with identical mixing.
