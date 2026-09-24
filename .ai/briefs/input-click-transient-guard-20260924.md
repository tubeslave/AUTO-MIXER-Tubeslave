# Preserve musical crests during input sanitation

## Thesis and observed failure

On HEAD 818f304f5, 19 continuous snare crests are classified as clicks solely
because their portions above 0.95 last less than 1 ms. Interpolating +/-3 ms
changes 5156 samples and removes 11.449668% of source energy. A -6 dB source
change suppresses all 19 detections, causing a -12.36 dB full-mix null residual
despite correct gain compensation. The prior fixed-sanitation experiment
localized the difference to this stage.

## Proposal

In backend/input_health.py, retain the existing full-scale candidate and context
checks, but require a short supported excursion and abrupt isolated boundaries
before permitting auto-repair. Check the excursion down to one quarter of its
height above the context; reject long support, incomplete boundaries, smooth shoulders and boundary
jumps not isolated from surrounding first differences. Keep repair spans and
all downstream safety/headroom logic unchanged. New tests cover smooth pulses,
damped musical attacks, polarity, sample rates and real recording glitches.

## Alternatives

Disabling all repair reintroduces the prior multichannel recording-glitch
failure. Lowering the absolute threshold addresses a different recall problem
and broadens false positives. Shortening interpolation alone still damages
wrongly labelled audio. A new learned detector is excessive for this bug.

## Scope and risks

Only input_health.py, its existing tests, canonical-run evidence and existing
autopilot surfaces. No dependencies, live writes or readiness promotion. The
stricter guard can miss smooth/band-limited clicks. A very short musical impulse
can be indistinguishable from a recording glitch; this is not a universal
classifier or listening approval. Keep these limits explicit.

## Validation and independent disagreement

Failing clean-transient regressions first, existing synthetic click regressions,
all 16 real source diagnostics, and control/-6 dB canonical renders on both
existing corpora. Verify known synchronized recording glitches remain detected
and the 16-track control remains unchanged. Standard suite after truth writes.
Unilateral shared-DSP patch pending Claude review; independent acceptance is
not claimed. Threshold calibration beyond these fixtures remains open.
