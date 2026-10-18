# STUDIO bounded task: Full-mix Compression Context Adapter v1

Base branch head when task started: `d24d7656772e191969a98309d037fa961f3358de`.

## Why this is next

Compression Iteration Bridge v1 explicitly leaves one blocker: source-level
Compression Director evidence is not evidence that a complete mix improved. The
highest-priority unblocked task is therefore to evaluate one source replacement in
an otherwise unchanged full-mix context and feed real before/after snapshots into
the existing Perceptual Critic and Autonomous Iteration policy.

## Bounded implementation

- Add one STUDIO-only `compression_mix_context` adapter.
- Require immutable stereo baseline/original/candidate contribution layouts.
- Active-RMS-match candidate contribution to the original before insertion.
- Replace exactly one routed source contribution; hash/reconstruct the unchanged rest.
- Update protected vocal/drum anchor only when that group owns the replaced source.
- Compute before/after whole-mix PerceptualSnapshot evidence.
- Append context failures to the existing compression objective veto path.
- Export a whole-mix A/B only for candidates whose existing transition reaches
  `human_listening`; never promote baseline here.

## Real evidence prepared in this run

Belye Stai, 20.0-40.0 s controlled context. Non-lead contributions are real
recipe-processed tracks from the existing local DSP run. The lead reference is raw
VALERA_VOX passed through one identical static HPF/EQ/pan/gain mapping; the candidate
is the existing Compression Director v1.1 balanced vocal audition passed through the
same mapping. Candidate contribution was then active-RMS matched to the original
routed contribution by -0.015049 dB before insertion. No FX, mix glue, mastering or
neural processing is used in this validation context.

Measured whole-mix PerceptualSnapshot deltas after that source match:
- vocal_intelligibility: +0.016682
- foreground_db: +0.146636 dB
- harshness: +0.015367
- density: +0.002500
- width_db: -0.059978 dB
- climax_lift_db: -0.678667 dB
- punch_db: 0.0 dB

Under the repository Perceptual Critic policy this candidate is rejected:
`target_not_improved` because +0.016682 < +0.020, and `climax_regression` because
0.678667 dB > the 0.50 dB protected limit. It therefore must not receive an audition
export from the adapter and must not become a baseline.

Replacement algebra remained bounded: baseline reconstruction max error was 0.0 and
candidate-rest max error was 2.9802322387695312e-08. This experiment is a controlled
real-source validation context, not a claim that the previously delivered Belye Stai
mix was rebuilt or improved.
