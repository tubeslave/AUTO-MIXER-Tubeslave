# STUDIO ADR: Belye Stai Bass Processor Adapter v1

## Decision

Add a source-bound adapter for the delivered Belye Stai bass chain so Compression Director experiments replace the **first existing bass compressor in place** instead of stacking an extra compressor after the processed bass.

The no-change path reproduces the frozen recipe:

`RAW -> HP/LP + EQ -> slow bass-band ride -> compressor 1 -> compressor 2 -> final static level`

For an experimental first-stage replacement:

- RAW, EQ, rider and topology stay frozen.
- The original second compressor **settings/threshold** stay frozen, but its gain-reduction envelope is recomputed from the new input.
- The original final static gain stays frozen.
- Audition loudness matching is external and must be reported separately.
- Kick ducking remains in the downstream full-session renderer and is not baked into this local adapter.
- No saturation, learned audio model, clipping, automatic winner, or baseline promotion is introduced here.

## Why bass needs its own path

The human-approved vocal compression result does not justify copying vocal timings or GR to bass. Bass has different event spacing, body consistency and low-frequency recovery requirements. The current generic Compression Director family was therefore treated only as a diagnostic probe.

On the full 207 s Belye Stai bass source, the frozen original adapter reproduced the delivered processed bass sample-for-sample. Three generic role-based first-compressor replacements then converged to their requested active-P95 GR targets but **all increased** fixed-event bass body-level spread versus the frozen current bass:

- current bass: 2.781 dB P90-P10 body spread
- preserve_transient: 4.626 dB
- balanced: 3.916 dB
- control: 3.238 dB

Those candidates are rejected as a basis for automatic bass improvement. This is useful negative evidence: the generic macro-timing formula yields releases that are too slow and target GR that is too light for this bass source. A bass-specific stability objective is required rather than reusing the vocal/generic objective.

Full-mix rerenders showed only very small changes in protected mix snapshots, so the rejection comes from the intended bass-specific diagnostic rather than a collateral full-mix regression.

## Acceptance boundary

The adapter implementation may be accepted after tests/CI. No replacement compressor candidate is musically accepted by this ADR. Human listening remains mandatory for any future surviving bass candidate.

## Next task

Build Bass Compression Director v1 around a bounded technical objective: stabilize note/body level while preserving attack-to-body contrast and recovery between events. Return an audition family, not a machine winner, and validate every candidate in the full routed mix before human A/B.
