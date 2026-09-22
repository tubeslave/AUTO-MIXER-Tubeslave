# Mixing Director v1.3: Space Director

Accepted baseline: v1.2 Dynamics + Masking/EQ.

Goal: create one coherent acoustic world without washing out the mix.

Depth hierarchy:
- closest: lead vocal, kick, bass;
- near/mid: snare, secondary vocal, toms;
- mid: guitar;
- farther: cymbals, keys, playback.

Architecture:
1. one deterministic common stereo room shared by the whole mix;
2. per-source send and pre-delay from depth role;
3. room low end is filtered to protect punch;
4. dense sections may gain at most ~1.6 dB more room send, not broadband level;
5. lead vocal remains comparatively dry and uses the longest pre-delay;
6. snare/toms may have a noticeable but short rolling room;
7. no algorithmic long vocal hall yet;
8. no new tonal EQ, saturation, limiting or reference matching;
9. Space Critic rejects low-end mud, excessive side-energy growth, hot wet bus or loudness cheating;
10. level-matched A/B against v1.2 is mandatory.

The room is a shared acoustic field, not eighteen unrelated reverbs.
