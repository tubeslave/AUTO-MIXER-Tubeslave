# ADR: separate causal compression from offline riding and makeup

## Context
Workbench conflates performance riding, frame compression and makeup. A mono-summed
detector cancels anti-phase L/R. Silence and short audio crash. Belye Stai has a
separate local DSP recipe; that does not fix the shared module.

## Options considered
Patch empty masks only; reuse live AutoCompressor; or implement a small stateful
feed-forward core and retain existing Workbench entry points.

## Decision and why this won
Choose a studio-only core. RMS averages channel power, peak uses maximum channel
magnitude. Both use one linked gain. A branching dB-domain GR envelope exposes
independent e-folding attack/release constants (63.2% per tau); RMS integration is
separate and affects the end-to-end response. HPF is detector-only; external
sidechain is supported. No lookahead, implicit makeup, rider or clipping in core.
The full-source offline adapter identifies the slow Gaussian rider separately,
uses a fixed RAW-active mask for bounded level matching and respects -1 dBTP
headroom before positive makeup. GR never includes ride or makeup. Diagnostics
remeasure PCM and label active 20 ms RMS spread separately from events/crest/LRA.
No extra mandatory dependency; optional Numba compilation has a tested Python fallback.

## Rejected alternatives
No threshold weakening, inferred musical acceptance, live-coupled implementation,
new plugin, neural processing or blind mass replacement of mastering dynamics.

## Implementation plan
Add mixing/compression.py; wire dynamics.apply/analyze_frames while retaining
profile fields and call shapes; add regression tests and raw Belye Stai audition.

## Test plan
Hard/soft knee, +12 dB at 4:1 => 9 dB steady GR; analytic attack/release on four
sample rates; no pre-onset GR; in-phase/anti-phase equivalence; linked stereo;
full vs irregular chunks including HPF and external sidechain; reset; silence;
partial frames; bypass; invalid PCM/config; measured PCM spread and bounded makeup.
Exact integration CI must pass Python 3.10/3.11/3.12 before merge.

## Risks and rollback
Correct causal compression changes sound versus the frame smoother. Profile defaults
and automatic threshold remain proposals, not an autonomous musical director.
Only new audition files are rendered here. Old mixes, recipes and live/backend are
untouched. Revert the studio patch to restore the legacy implementation. Code
acceptance is not listening acceptance; no audio baseline may be promoted here.

## Source grounding
Giannoulis, Massberg, Reiss: Digital Dynamic Range Compressor Design (JAES 2012)
and Parameter Automation (JAES 2013): feed-forward gain computer, soft knee,
distinct detector, ballistics and makeup; no numerical genre presets copied.
https://joshreiss.github.io/documents/2013/Giannoulis%20Massberg%20Reiss%20-%20dynamic%20range%20compression%20automation%20-%20JAES%202013.pdf
