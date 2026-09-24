# Verify offline drum phase decisions

## Thesis and understanding

The actual offline close-mic stage fails three signal-level regressions on
b07d5d630: unrelated sources receive 7.397 ms delay; inconsistent 3/8 ms
arrivals overwrite the previous state; a previous 5 ms delay cannot converge
to the measured 3 ms absolute target. Ten positive/control cases pass.
This is an audible-signal decision bug, not a request for another control layer.

## Proposal

Reuse phase_v3 onset detection, minimum hit count, MAD/IQR and polarity
thresholds. Keep GCCPHATAnalyzer's documented reference/target order and use
sample-rate-sized padded hit windows. Require PSR and time-domain correlation,
consistent lag/polarity across hits, and non-decreasing low-band sum energy
on the same passages with the actual zero-filled delay. Stage the absolute
delay and polarity locally; mutate ChannelPlan only after verification.
Record applied/unchanged/skipped decisions with reasons.

## Files and alternatives

tools/offline_agent_mix.py, tests/test_offline_agent_mix_phase.py, scoped
evidence/ADR and existing autopilot surfaces. No new dependency/module or live
surface. Disabling phase universally loses verified positive recovery; gating
on GCC confidence alone is invalid because PHAT peak normalization depends on
raw signal energy. The shared phase_v3 estimator's fixed 4096 frame also misses
44.1 kHz 90 ms windows; do not import that estimator unchanged.

## Risks and tests

Conservative onset/correlation checks may abstain on busy recordings. Correlated
bleed and broadband-vs-low-band disagreements require explicit negative tests;
low-band sum is a limited analysis check, not listening or full-mix approval.
Run sample-rate/polarity/gain controls, insufficient/inconsistent/unrelated
sources, zero-filled boundaries, cap handling and repeat convergence. Then
run canonical full renders on both existing real corpora and the standard
suite after truth writes. Third independent corpus and listening remain open.

## Independent disagreement

Unilateral offline bug fix under ops/collab/PROTOCOL.md, pending independent
DSP review. A reviewer may prefer band-specific onset/polarity estimation;
do not claim independent agreement or live authority from this patch.
