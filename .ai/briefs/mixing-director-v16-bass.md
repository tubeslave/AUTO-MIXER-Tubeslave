# Mixing Director v1.6: Bass Director

Accepted baseline: v1.5 Vocal Director.

Goal: powerful, even bass that stays behind the vocal/drums, avoids excessive midrange, and remains audible on small systems.

Architecture:
1. preserve accepted general Dynamics and Masking stages;
2. derive a kick sidechain from the reinforced kick, focused around 45–110 Hz;
3. duck bass only around kick events, max ~1.4 dB, instead of broadband permanent bass reduction;
4. dynamically control excessive 250–900 Hz bass prominence, max ~1.5 dB;
5. add a very small level-neutral nonlinear harmonic layer for small-speaker translation;
6. no sub boost by default;
7. no static smile-EQ by default;
8. Bass Critic checks total bass level, kick-band relationship, midrange loss and 700 Hz–2.5 kHz translation band;
9. preserve v1.3 shared room and all accepted Instrument Directors;
10. level-matched A/B against v1.5 is mandatory.
