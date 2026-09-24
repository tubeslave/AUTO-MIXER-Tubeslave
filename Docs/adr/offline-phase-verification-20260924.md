# Offline drum phase verification

## Context

Three reproducible failures in the real offline phase function contradict
safe correction and convergence. Full drum renders previously used delays
with GCC confidence 0.162--0.244; that statistic alone is not calibrated proof.

## Options considered

Disable phase, impose a confidence threshold, or verify independent hits and
the actual candidate before applying it.

## Decision

Verify onset-window PSR, minimum hit count, MAD/IQR, aligned correlation and
polarity consistency using existing phase_v3 rules. Verify low-band sum energy
on fixed windows before committing absolute delay/polarity to ChannelPlan.
Reject late targets requiring advance, boundary estimates and weak evidence.
Leave overhead stereo timing unchanged. Emit explicit abstention reasons.

## Why this won

Preserves positive correction without treating an arbitrary correlation peak
as authority. Absolute targets fix repeated-run overshoot; zero-filled delay
matches the renderer, unlike circular roll.

## Rejected alternatives

Do not modify shared GCC confidence normalization or add another phase engine.
Do not use phase_v3's fixed-size delay estimator without repairing its
sample-rate boundary. No universal disable and no negative-only lag selection.

## Implementation plan

Change only the existing offline function and its imports; add direct
signal-level regressions, then render both available real corpora. No console
I/O, no safety-gate or maturity promotion. Independent review remains pending.

## Test plan

44.1/48/96 kHz known lag with both polarities, repeat convergence, gain scaling,
unrelated/inconsistent hits, insufficient hits, silence, caps, low-band veto,
and canonical end-to-end render controls. Full repository suite follows final
truth updates; report CP05 failures without deselection or false attestation.

## Risks and rollback

Onset selection and conservative correlation can abstain on valid acoustic
material; the evidence covers analysis sums, not a listening verdict. Preserve
the previous ChannelPlan on every rejected candidate. If needed revert this
scoped commit; inputs and console state are never changed. Hardware/readback/
rollback proof and independent quality evaluation remain external gates.
