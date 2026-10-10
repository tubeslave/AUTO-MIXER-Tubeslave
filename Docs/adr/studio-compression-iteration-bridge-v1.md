# ADR: Compression Director v1.1 as a bounded Autonomous Iteration family

## Context
Compression Director v1.1 can produce three calibrated compressor candidates, but until this change it is not wired to the existing causal plan, Perceptual Critic and Autonomous Iteration decision path. Keeping a separate compression-specific acceptance policy would duplicate safety logic and create a route for accidental machine-only baseline promotion.

## Decision
Add one STUDIO-only bridge which:

1. asks Compression Director v1.1 for exactly `preserve_transient`, `balanced`, and `control`;
2. creates the normal causal plan through `causal.make_plan`, retaining exactly one `bypass/no_change` counterfactual;
3. renders the accepted causal `LinkedCompressor` and records source/candidate PCM hashes, frames/sample rate, calibrated active-p95 GR, max GR, timing-censor evidence and true peak;
4. creates a fair audition plan with exact fixed-mask active-RMS match followed by one common trim for headroom;
5. sends mix-context before/after snapshots through the existing `mixing.perceptual_critic.accept_candidate` and then `autonomous_loop.resolve_candidate_iteration`;
6. treats an objective rendering/calibration/headroom failure as a veto and rollback;
7. marks every compression candidate as requiring human listening and forbids this bridge from promoting a baseline.

The Perceptual Critic still evaluates a **mix-context** hypothesis. Source-level Compression Director evidence is not treated as proof that the mix improved.

## Why this option
It reuses the already tested acceptance path rather than adding a second policy. The Director remains a hypothesis generator, Perceptual Critic remains an engineering comparison gate, Autonomous Iteration remains the transition authority, and human listening remains the subjective authority.

## Rejected alternatives
- Pick the candidate closest to target GR. That only proves calibration, not musical quality.
- Rank candidates by source-only crest/RMS statistics. Those do not establish mix-context improvement.
- Auto-promote a machine-safe Perceptual Critic result. All audible compression candidates are subjective and stay human-gated.
- Put a limiter into the A/B matcher. The current common-trim plan avoids changing the comparison with an extra nonlinear process.

## Acceptance tests
- exact three named compressor variants plus one no-change counterfactual;
- immutable source hash and identical frames/sample rate for candidate renders;
- JSON-safe auditable evidence;
- machine-safe perceptual evidence routes to `pending_human_review` with baseline unchanged;
- objective gate failure overrides an improved perceptual proxy and routes to rollback;
- full repository CI must pass before merge.

## Evidence boundary
Existing real-source grounding is the 60-second Belye Stai v1.1 experiment: all 9 source candidates met their calibrated active-p95 GR targets within +/-0.08 dB. That experiment did **not** establish a musical winner and is not reinterpreted here as one. A future full-mix experiment must insert candidates in real mix context, produce matched auditions and wait for human listening before baseline promotion.
