# Mixing Director v1.8: Autonomous Iteration Loop

Accepted baseline: v1.7 Guitar Director.

The loop changes the architecture from a fixed processor chain into bounded engineering iteration.

Each iteration:
1. analyze the current render globally and by musical sections;
2. Mix Critic emits diagnostics, not taste scores;
3. choose exactly one highest-confidence hypothesis;
4. make one bounded change, normally <=0.8 dB;
5. render candidate B;
6. compare target metric plus collateral guardrails;
7. accept or roll back;
8. continue until no high-confidence hypothesis remains or iteration budget is exhausted.

Hard rules:
- no more than 4 iterations per pass;
- no simultaneous unrelated changes;
- no loudness cheating;
- no broad tonal change >0.45 dB without a dedicated tonal hypothesis;
- preserve >=1 dB peak headroom;
- every accepted iteration records before/after metrics and exact parameter delta;
- every rejected iteration is logged;
- final human level-matched A/B remains mandatory during development.

The critic informs decisions but does not pretend objective metrics fully encode musical quality.
