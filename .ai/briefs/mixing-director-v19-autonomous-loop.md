# Mixing Director v1.9: Autonomous Iteration Loop

Baseline is now the complete instrument-aware v1.8 chain:
Context -> Balance -> Section Automation -> Dynamics -> Masking/EQ -> Space ->
Drum -> Vocal -> Bass -> Guitar -> Keys/Playback.

The loop must not compensate for a missing Instrument Director anymore.

Per iteration:
1. diagnose current render globally and by active musical sections;
2. choose exactly one high-confidence hypothesis;
3. make one bounded change, normally <=0.8 dB;
4. render B;
5. verify the target metric improved;
6. reject loudness cheating, headroom loss or broad tonal collateral damage;
7. accept or roll back;
8. stop after four iterations or when only low-confidence issues remain.

During development, every accepted final loop render must still be human-reviewed level-matched against v1.8.
