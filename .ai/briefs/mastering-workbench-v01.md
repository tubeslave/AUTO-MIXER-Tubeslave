# Mastering Workbench v0.1

Goal: an autonomous mastering decision layer inspired by modern mastering workflows, not a reverse-engineered Ozone implementation.

Initial chain:
Analyzer -> self-relative Stabilizer -> adaptive Clarity -> Impact peak conditioning -> oversampled soft clipper -> four-band Maximizer -> regression report.

Rules:
- Ozone/reference products are design inspiration only. Do not claim algorithmic equivalence.
- Stabilizer is self-relative by default; references/genre targets are optional priors later.
- Clarity reduces local spectral protrusions, not broad tonal balance.
- Impact protects useful dynamics and conditions expensive peaks before loudness.
- Clipper is bounded and oversampled.
- Maximizer uses four independent band envelopes to reduce cross-band pumping, inspired by the public architectural description of IRC 5, but contains no proprietary implementation.
- Every module emits measurements and delta/regression metadata.
- Future: Bass Director, Stereo Director, exciter, true-peak oversampled detector, LUFS targeting, codec preview, automatic module bypass and level-matched acceptance tests.
