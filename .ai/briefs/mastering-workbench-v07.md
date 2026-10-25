# Mastering Workbench v0.7: cumulative micro-improvements

A/B audibility of one bounded mastering delta can be below the listening threshold while
several independent deltas may become meaningful together.

Policy:
1. Never call an inconclusive micro-change an improvement.
2. Technical survivors may be accumulated only when they affect different controls.
3. Re-render the compound candidate from the premaster. Do not serially process already mastered files.
4. Run the Musical Regression Critic again on the compound result. Individual PASS does not imply compound PASS.
5. Compare the compound candidate against the Quality Baseline level-matched.
6. If the compound is still indistinguishable, stop that branch.
7. If it is audibly preferred and passes regression, it may replace the Quality Baseline.

For the current experiment, combine the least-regressive Clarity/Impact/Clip changes while
keeping Stabilizer and Bass Director at the v0.3 baseline. Do not include Stabilizer Half
because it failed snare-transient regression.
