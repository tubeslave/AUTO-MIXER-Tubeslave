# Editing Director v0.7: multi-mic drum performance

Drum hit level editing must not rely on one close mic.

For kick/snare:
1. detect a candidate on the primary close mic;
2. confirm temporal evidence on the secondary close mic and/or overheads;
3. assign a multi-mic confidence score;
4. measure event level on the primary mic;
5. compare only against nearby confirmed hits in the same broad song section;
6. require both >=4 dB deviation and robust |z| >=2.8;
7. apply only 45% correction, hard-capped at +/-1.5 dB;
8. when an event is edited, apply the same smooth gain move to all drum microphones to preserve phase and room perspective;
9. do not normalize fills, accents, ghost notes, or unconfirmed events;
10. render identical mix and level-matched A/B against v0.6.
