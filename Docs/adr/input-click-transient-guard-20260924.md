# Guard click repair against continuous musical crests

## Context

The existing detector measures only duration above an absolute 0.95 threshold.
This labels continuous snare crests as isolated clicks; the default repair then
removes 11.45% of source energy on a real test track. Prior full-render trials
isolated this stage as the cause of gain-dependent differences.

## Options considered

Disable repair; change the amplitude threshold; shorten interpolation; add a
conservative short-excursion/discontinuity requirement to the existing detector.

## Decision

Require supported duration and both discontinuities before auto-repair, keeping
existing full-scale candidate selection, context checks and repair behavior.
Use quarter-excursion support above the contextual 95th-percentile amplitude,
quarter-peak minimum jumps, and 12 dB edge contrast against the 95th percentile
of local absolute-envelope differences. Both the wide context and immediately
adjacent one-millisecond cycles are checked. These are conservative
initial guards, not externally validated perceptual thresholds.

## Why this won

This directly tests the missing distinction: a brief crest above threshold is
not necessarily a brief discontinuity. It only rejects previously accepted
candidates and preserves known recording glitches in the validation corpus.

## Rejected alternatives

No blanket bypass of sanitation; no broad gain-invariant click detector; no
new learned dependencies; no retuning unrelated phase, balance or dynamics.

## Implementation plan

Write failing smooth-transient tests; add bounded support/edge checks; repeat
synthetic tests, real corpus diagnostics and canonical full audio renders.

## Test plan

Several sample rates and polarities, smooth pulses and damped attacks, true
impulses, real source hash checks, true peak and full-mix null comparison. Run
the standard pytest command after final truth writes. Independent review and
source-bound CP05 remain separate requirements.

## Risks and rollback

Band-limited, long or boundary-truncated clicks may now be left untouched.
Sub-full-scale clicks remain outside the existing default candidate threshold.
Impulsive musical events remain ambiguous; listening and broader calibration
are required. Revert only this detector patch if regression is demonstrated;
never overwrite source files, disable headroom checks or contact the console.
