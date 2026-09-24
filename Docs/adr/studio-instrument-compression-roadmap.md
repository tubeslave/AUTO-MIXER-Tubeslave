# ADR: instrument-specific compression, not a copied vocal preset

## Evidence motivating the change
The user accepted the Belye Stai v2 vocal for stable level and mix position. The previous intelligibility-ratio target rejected that same candidate. Human preference and objective evidence are separate; record both. Do not redefine a threshold after seeing candidate scores to force acceptance.

## Shared mechanism, different musical questions
Use the tested causal linked compressor, measured active-P95 GR calibration, fixed reference event windows, immutable source identity, full routed rerender and level-matched listening. Attack/release/ratio are proposals based on each source, never inherited from the vocal success. No-change is a valid outcome.

| Stage | Musical objective to validate | Protect | Status |
|---|---|---|---|
| Bass | consistency of note bodies, sustained foundation, controlled peaks | note attack, low-frequency waveform, deliberate accents, kick relationship | first in-place adapter and real experiment |
| Kick + snare | control body/peaks independently of leading attack | synchronized mic group, ghost notes, inter-hit recovery, cymbal bleed | next adapters, not implemented here |
| Toms | stable fill level and decay | attack, tails, no extra bleed | planned |
| Guitar | stable rhythmic layer or lead sustain appropriate to role | pick articulation and accents; no forced compression of already-limited distortion | planned |
| Keys/playback | context-dependent consistency | stereo link, piano transient vs pad sustain, original programmed dynamics | planned |
| OH/cymbals | investigate only demonstrated level problems | stereo image, cymbal decay, no unnecessary pumping | no-change first, planned |

## First bounded implementation
Belye Stai bass RAW -> 33 Hz/6 kHz filters -> original 175/850 Hz EQ -> original 50-1100 Hz slow rider -> compressor 1 -> compressor 2 -> level. No-override matches the original chain. Replacement uses bass-role proposals before compressor 1, with original compressor-2 threshold and final gain stored in source-bound controls. The second stage recomputes gain reduction, but not its parameters. This avoids an extra compressor and a hidden retune of downstream controls. Explicit A/B matching can apply a separate measured trim; never label it compressor GR.

Frozen arithmetic helpers are imported from the unchanged song vocal adapter to preserve exact arithmetic. This is deliberate reuse of DSP implementation, not vocal configuration. A future common frozen-DSP extraction must prove both adapters' invariance before moving code.

## Validation
Static contract tests plus independently rerun original procedural bass; complete session no-change using human-preferred v2 vocal; three full-song replacements; remeasure sources and whole mix. Diagnostic macro-envelope event windows are NOT note transcription or a music-quality score. Safe GR and reduced dynamic spread alone cannot select a winner.

## Boundaries
Do not modify live DSP, mastering, the accepted vocal, existing compressor core or critic thresholds in this task. No neural processing or external paid credits. Existing masters remain immutable. Scoped human acceptance is saved; historical target failure is not erased and no automatic baseline promotion is performed.
