# Mix-bus spectral reference recovery regression

Scope: DEC-0008 recovery validation; offline DSP, no console contact.

The production spectral analyzer and offline EQ renderer were exercised through
`recovery_trial.run_trial`, with only damaged audio passed to the repair function.
Three deterministic pink-noise signals (seeds 7, 23, 91; 4 s at 48 kHz) each
received the six existing defects. A -4 dB peaking defect at 1 kHz caused three
unrelated cuts and approximately 19% additional spectral error.

The octave around the reference frequency anchors the entire target. A local
deficit at that anchor lowers the whole target. Proposed fix: for mix_bus only,
estimate the target intercept from the median slope-compensated spectrum over
80 Hz to 14 kHz. Retain existing source-role anchors and all action bounds.

Alternatives: no-op all master correction loses useful cuts; stronger correction
amplifies the error; fitting the measured slope would erase the intended profile.
Risk: changed mix-bus proposals on real music; requires separate multitrack and
listening validation. Three noise seeds are not three multitracks.

Tests: known scoop must not worsen; original noise must remain unchanged;
mud/harshness/boxiness must still improve; absolute input gain must not alter
decisions; existing spectral and recovery tests; standard repository suite.

Files: `backend/heuristics/spectral_ceiling_eq.py`,
`tests/test_spectral_ceiling_eq.py`, this brief, ADR and diagnostic report.
This is a unilateral cross-area DSP fix for subsequent owner review under
`ops/collab/PROTOCOL.md`; no independent review has yet occurred.
