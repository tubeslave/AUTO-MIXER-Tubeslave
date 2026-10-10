# STUDIO: Belye Stai Session Renderer Adapter v1

## Goal
Convert the delivered `Belye Stai` procedural DSP recipe into a deterministic reusable `SessionRender` callback without changing the accepted delivered premaster or claiming a new musical result.

## Why now
`studio-routed-contribution-renderer-v1` proved that compression candidates must be evaluated through a full routed session rerender. Its recorded blocker is that `Belye_Stai_DSP_Recipe_and_Reports.zip/code/mix_dsp.py` is procedural, path-bound and not callable through that contract.

## Bounded task
Expose the frozen **post-local-track-processing / pre-context** stage as an immutable override boundary and rerun every downstream dependency from there:

- DRUMS group sum/compression/level;
- lead-vocal-driven guitar lift and 1.4-4.2 kHz masking;
- kick-linked bass low-band ducking;
- drum room, vocal chamber and filtered slap;
- shared MIX_GLUE gain reduction;
- common edge fade and trim;
- final bounded 159.056 s drum-accent refinement.

Do not migrate raw-track local EQ/dynamics in this task. Do not change the recipe constants. Do not master a new file or promote any audio baseline.

## Acceptance evidence
1. Synthetic tests prove exact no-change override identity, downstream dependency rerendering, fail-closed override validation and deterministic PCM24 export.
2. On the actual 44.1 kHz / 9,128,700-frame project, regenerate the processed-track stage from the two original source ZIPs and frozen recipe, render through the new adapter, apply the frozen final PCM24 dither, and compare against the already-delivered `Belye_Stai_Premaster_44k24.wav`.
3. Acceptance target for the real-song reproduction is sample identity of decoded PCM24 and file SHA-256 identity. If that fails, do not weaken the target silently; record the exact discrepancy and revise or stop.
4. Full repository CI must pass before merge.

## Safety / scope
- STUDIO/offline only.
- Conventional DSP only; no learned audio model inference.
- No live, OSC or hardware control changes.
- No paid services.
- No subjective audio acceptance. Human listening remains mandatory before any future compression candidate can replace a musical baseline.

## Review availability
No independent Kimi reviewer is available in this run. Self-review must be labelled as such and cannot be represented as independent review.
