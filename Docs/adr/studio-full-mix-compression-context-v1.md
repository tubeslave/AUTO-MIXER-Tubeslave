# ADR: Full-mix Compression Context Adapter v1

## Context

Compression Director v1.1 and Compression Iteration Bridge v1 validate source-level
candidate calibration, render safety, causal evidence and baseline policy. They do
not prove that a source-level compression change improves the complete mix. A valid
source candidate can change masking, foreground balance, width/density proxies or
section dynamics once it is reinserted beside the unchanged instruments.

## Decision

Add one STUDIO-only context adapter that accepts an immutable stereo baseline mix,
its exact already-routed source contribution, and the candidate contribution in the
same routing convention. Before insertion the candidate contribution is matched to
the original contribution with the existing fixed-reference-active RMS method. The
adapter then computes exactly:

`candidate_mix = baseline_mix - original_contribution + matched_candidate_contribution`

It records the hashes and reconstruction error, updates the vocal or drum anchor
when the replaced source belongs to that protected group, derives before/after
PerceptualSnapshot evidence from the complete mix, and delegates acceptance to the
existing Compression Iteration Bridge -> Perceptual Critic -> Autonomous Iteration
path. Context failures are appended as objective vetoes rather than creating a
second policy.

A full-mix level-matched audition pair is returned only when the existing transition
says `human_listening`. Rejected candidates produce no audition render. This module
cannot promote a baseline.

## Why

This closes the causal gap between “the compressor hit its requested GR” and “the
mix got better”. It also prevents a candidate from winning merely by becoming louder.
The rest of the mix is reused unchanged, so the experiment varies exactly one source
contribution.

## Boundaries

- STUDIO/offline only. No live, OSC or backend path imports this adapter.
- Input contributions are already routed stereo contributions. Pan/EQ/send topology
  belongs upstream and must be the same for original and candidate.
- No limiter or mastering is inserted into the evaluation path.
- The adapter does not rank candidates. It evaluates one named candidate at a time.
- Human listening remains mandatory for subjective acceptance.
- A source-dependent effect return must be included in the supplied contribution or
  supplied through an explicitly updated context; the adapter does not guess send
  ownership from names.

## Validation plan

1. Exact one-source replacement and stable “rest of mix” evidence.
2. Gain-only candidate collapses to no-change after source matching and is rejected.
3. A machine-safe whole-mix candidate can only reach pending human review and may
   export an audition pair, never promote the baseline.
4. Vocal/drum replacements require the corresponding anchor bus and update it with
   the same replacement.
5. Invalid layout/timeline fails closed.
6. Real Belye Stai experiment: one v1.1 vocal candidate is reinserted into a
   controlled 20-second real-source context and judged by the exact Perceptual
   Critic thresholds. Source-level calibration is not treated as mix acceptance.

## Rollback

Revert the adapter/test/state commits. Existing compression DSP, Director,
Iteration Bridge, delivered mixes and mastering remain unchanged.
