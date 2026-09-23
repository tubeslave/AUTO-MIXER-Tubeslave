# Canonical offline renderer cannot finish its CLI result

Goal: run the full pipeline on real multitracks before evaluating quality.
Pre-fix evidence: `/tmp/automixer-goal-corpus-20260923.iXSgVS/drums.pipeline.log`.
The eight-track, 26.36-second drum render creates WAV and reports, then exits 1
because `source_envelopes.*.crest_ok` is a NumPy bool, which JSON cannot encode.

Decision: normalize the boolean at the existing manifest DTO boundary in
`tools/repa_song_soundcheck_pipeline.py`. DSP values and audio remain unchanged.
Alternative: global JSON default=str would hide type errors and is rejected.
Test real envelope extraction from silent, tonal and noisy WAVs, then rerun
the canonical CLI with explicit outputs on the two available multitracks.
The third multitrack and accepted reference mixes have been requested; local
renders establish execution/repeatability only, not listening acceptance.

Scope: one DTO field plus focused regression. Full suite remains subject to
the separately unbound CP05 gate from the previous source checkpoint.
No console contact, live authority, source-file edits or new dependencies.
