# Editing Director v0.8: Source Cleanup / De-bleed / De-noise

Problem observed in the Ptitsa multitrack:
- close drum mics contain substantial bleed;
- vocal mics contain substantial stage/instrument bleed;
- bass and guitar contain noise plus fret/scrape/inter-note artifacts;
- this contamination degrades downstream balance, compression, masking and autonomous decisions.

Hard rule: DO NOT solve this with a binary gate.

Pipeline:
1. classify source and active musical regions;
2. estimate stationary noise floor from non-musical regions;
3. estimate cross-mic bleed using other synchronized microphones and spectral coherence;
4. build continuous time-frequency masks (Wiener/ratio style), bounded by source role;
5. preserve target transients by protecting high-confidence onset bins;
6. for vocals, protect consonants, breaths inside phrases and phrase tails;
7. for drum close mics, attenuate only bleed-dominant time-frequency regions; never mute between hits;
8. overheads receive minimal cleanup because they intentionally contain the whole kit;
9. guitar/bass use a separate inter-note artifact detector: high-frequency scrape/noise + low harmonic confidence + location between stable notes;
10. artifact attenuation is continuous and crossfaded, not a gate;
11. no phase-changing multiband reconstruction across paired drum mics unless identical latency and phase QA pass;
12. Cleanup Critic compares raw vs cleaned source for transient loss, phase relationship, active-note loss, new clicks and actual artifact reduction;
13. render raw/cleaned solo A/B plus full-mix A/B before accepting;
14. after acceptance, Cleanup moves near the START of Editing Workbench, before pitch, performance consistency, Dynamics and Mixing.

Recommended order:
Raw -> phase/alignment analysis -> Cleanup -> groove/performance editing -> vocal pitch -> final Editing QA -> Mixing Director.

This stage should improve later AI decisions because compressors and critics will see the intended source rather than stage contamination.
