# Mastering Workbench v0.2

Adds a second autonomous tier after v0.1 validated as a small subjective improvement.

New modules:
- Bass Director: low-end sustain/punch diagnosis and bounded dynamic cleanup.
- Stereo Director: conservative M/S width changes above 180 Hz only, guarded by correlation.
- Exciter: subtle parallel high-mid harmonic density.
- Decision policy: modules are conditional; a mastering chain is proposed from analysis instead of always enabling everything.
- Acceptance gate: reject excessive crest loss, stereo collapse/over-expansion, or negative correlation.

v0.2 principle: do less by default. A module must have a diagnosed reason to exist.
