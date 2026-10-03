# ADR: event-aware Compression Director candidates without autonomous music acceptance

## Decision
Keep the accepted causal linked compressor unchanged and add an opt-in director above it. The director measures a role-smoothed macro RMS envelope, estimates 6 dB attack/recovery timing and inter-event spacing, then emits three bounded candidates: `preserve_transient`, `balanced`, and `control`.

These labels describe intended intervention strength, not a quality ranking. All candidates set `requires_human_review`, `requires_human_listening`, and `baseline_eligible=false`.

## Why
The previous role defaults were safe starting points but did not use the actual source's temporal structure. A naive peak finder was rejected during prototyping because it counted thousands of bleed/waveform-scale ripples on real Belye Stai sources and produced implausibly short releases. The accepted design uses role-specific envelope smoothing, prominence and minimum event spacing to measure macro behavior instead.

## Level matching and headroom
Compression evaluation must not confuse loudness with quality. `level_match_evidence` uses a fixed reference-active 20 ms RMS mask and reports the exact candidate gain needed for internal-float matching. An optional true-peak ceiling is treated as a named export constraint and any residual mismatch is exposed.

For listening A/B, `audition_pair_plan` first computes exact active-RMS matching and only then applies one common trim to both reference and candidate so both fit the requested true-peak ceiling. It introduces no limiter or clipper and cannot make one side louder merely because of headroom.

## Rejected alternatives
- Replacing the existing default role profiles automatically: too large a behavioral change before listening evidence.
- Selecting the candidate with the lowest dynamic spread, most GR, or a single critic score: those metrics do not establish musical preference.
- Enforcing an arbitrary source-level -1 dBTP ceiling while judging track compression: it can create a false level mismatch inside a float mix pipeline.
- Neural or paid external parameter selection: unnecessary for this bounded task.

## Risks
Macro event detection is not note transcription and can still misread bleed or sustained sources. Role bounds remain hand-authored safety limits. Real musical acceptance therefore remains a matched human A/B decision.

## Rollback
The director is opt-in. Removing `audio_workbench/mixing/compression_director.py` and its tests restores the previous behavior; no existing mix/render path changes by default.
