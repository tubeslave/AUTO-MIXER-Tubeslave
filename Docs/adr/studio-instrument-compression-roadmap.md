# ADR: instrument-specific compression, not a copied vocal preset

## Evidence motivating the change
The user accepted the Belye Stai v2 vocal for stable level and mix position. The previous intelligibility-ratio target rejected that same candidate. Human preference and objective evidence are separate; record both. Do not redefine a threshold after seeing candidate scores to force acceptance.

## Shared mechanism, different musical questions
Use the tested causal linked compressor, measured active-P95 GR calibration where appropriate, fixed reference event windows, immutable source identity, full routed rerender and level-matched listening. Attack/release/ratio are proposals based on each source, never inherited from the vocal success. No-change is a valid outcome. A technical survivor is permission for human A/B only, never an automatic musical winner or audio-baseline promotion.

| Stage | Musical objective to validate | Protect | Status |
|---|---|---|---|
| Bass | consistency of note bodies, sustained foundation, controlled peaks | note attack, low-frequency waveform, deliberate accents, kick relationship | exact in-place adapter + fail-closed stability gate accepted; timing-only candidates produced no full-song survivor, so current compressor remains baseline |
| Kick In/Out | more consistent body without blunting front edge | synchronized mic group, quiet hits, inter-hit floor, downstream bass relationship | grouped adapter + gate accepted; `tighter_body` is a technical A/B survivor only |
| Snare Top/Bottom | stable body while retaining crack and ghost notes | bottom polarity/processing, quiet hits, inter-hit bleed, attack/body contrast | grouped adapter + gate accepted; `tighter_body` is a technical A/B survivor only |
| Toms | stable fill level and decay | attack, body→tail shape, quiet hits, no extra bleed | TOM_1/TOM_2/FLOOR adapters + gate accepted; common `preserve_attack` survivor is human-A/B only |
| Guitar | stable rhythmic layer or lead sustain appropriate to role | pick articulation and accents; no forced compression of already-limited distortion | exact local adapter + no-change-first actionability gate implemented; full-song GTR local spread 2.6685 dB is below the predeclared 2.8 dB action threshold, so zero candidates and current compressor retained; PR CI pending |
| Keys/playback | context-dependent consistency | stereo link, piano transient vs pad sustain, original programmed dynamics | **next**: separate source roles and prove an actionable problem before compression |
| OH/cymbals | investigate only demonstrated level problems | stereo image, cymbal decay, no unnecessary pumping | no-change first, planned |

## Implemented source-bound boundaries
Belye Stai bass, Kick In/Out, Snare Top/Bottom, TOM_1/TOM_2/FLOOR and GTR now have exact no-change processor boundaries around their original compressor locations. Replacement candidates are inserted at the original compressor stage, never stacked after the already processed source. Source-bound downstream gains and other frozen controls cannot be silently re-estimated. Full-session rerender is mandatory for a surviving local candidate.

Frozen arithmetic helpers are reused only to preserve the exact delivered DSP arithmetic. This is implementation reuse, not configuration transfer between musical roles. A future common frozen-DSP extraction must prove all affected adapters' no-change invariance before moving code.

## Validation pattern established by bass/drums/guitar
1. Reproduce the delivered local processor sample-for-sample.
2. Detect source events or active windows once from a fixed pre-compression reference and reuse identical evidence definitions for baseline and candidates.
3. First prove that the baseline has a role-specific actionable problem. If not, stop at `no_change`.
4. Change only predeclared compressor dimensions; do not weaken thresholds after seeing results.
5. Reject local candidates that fail the role-specific target or protected transient/bleed/decay/macro-dynamics metrics.
6. Rerender any survivor through the complete routed session with the accepted vocal frozen at its approved local insert.
7. Apply Perceptual Critic only as a protected-regression layer unless a role-specific perceptual target has actually been calibrated.
8. Export level-matched A/B for human listening. Code/CI success never constitutes musical acceptance.

Diagnostic event metrics are not note transcription or a music-quality score. Safe GR and reduced dynamic spread alone cannot select a winner. The guitar gate additionally demonstrates that *not changing* an already self-limited source is a successful autonomous decision.

## Boundaries
Do not modify live DSP, the accepted vocal, mastering, existing compressor core or critic thresholds merely to make an experiment pass. No neural processing or external paid credits. Existing masters remain immutable unless a human explicitly accepts a new audio version. Historical failures stay recorded; no automatic audio-baseline promotion is performed.
