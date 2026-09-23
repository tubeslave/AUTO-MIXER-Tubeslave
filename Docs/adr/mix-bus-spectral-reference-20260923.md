# Mix-bus spectral reference

## Context

DEC-0008 requires measured recovery, not oracle-only tests. A reproducible
1 kHz scoop makes the existing mix-bus analyzer cut three healthy regions.
The clean reference is used only by the evaluator, never by the analyzer.

## Options considered

- Keep the single-octave intercept: reproduces the regression.
- Disable mix-bus spectral correction: discards useful correction of excesses.
- Use a robust broadband intercept with the configured target slope.

## Decision

Use the third option for mix_bus. Source-specific profiles retain their current
anchor. No correction bounds, confidence requirements or console permissions change.

## Why this won

A local trough no longer shifts the whole ceiling. The median ignores localized
extremes while preserving the target slope and invariance to input gain.

## Rejected alternatives

Do not increase EQ strength, expose the injected defect to the analyzer or fit
the target slope to the damaged signal. Those would conceal the failure.

## Implementation plan

Extend the existing reference helper with an optional slope-aware broadband
estimate; use the existing 80 Hz to 14 kHz tilt measurement range. Opt in only
for mix_bus. Exercise the actual analyzer and offline EQ renderer in tests.

## Test plan

Run six recovery defects on three fixed noise seeds, with clean-audio controls
and gain invariance. Retain existing spectral tests and run the standard suite.
These are diagnostic signals, not accepted listening or multitrack evidence.

## Risks and rollback

Real-music changes remain unvalidated. Revert this source patch to restore the
previous anchor. No console state needs restoring. Independent review is pending;
this ADR records a local bug-fix decision, not new live authority.
