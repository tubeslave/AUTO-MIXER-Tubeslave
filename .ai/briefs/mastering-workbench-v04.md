# Mastering Workbench v0.4: loudness search

Do not target a genre LUFS number by default.

Search increasing Maximizer drive in bounded steps. For each candidate:
1. render the same representative windows;
2. loudness-match for quality comparison;
3. measure crest loss, stereo shift, band GR and final limiter GR;
4. stop at the first regression boundary;
5. choose the last candidate before that boundary;
6. full-song render only the chosen setting;
7. keep human listening as the final acceptance gate.

This is a guardrail search, not proof of perceptual optimality.
