<!-- source_type: research -->
# Diff2Mix, Sequential Stem Blending, and DiffVox integration notes

- discovered_at: 2026-09-10
- status: candidate / source-grounded
- auto_apply: false

## Diff2Mix

Source: https://arxiv.org/abs/2608.05442

Use the reference as a style prior while keeping the mix represented by explicit, bounded, editable DSP parameters. Do not optimize only a reference embedding distance. A candidate is accepted only after level-matched A/B and context/section critic checks.

## Sequential Stem Blending

Source: https://arxiv.org/abs/2608.05506

Evaluate each incoming stem against the current growing submix. A practical initial role order for rock is drums -> bass -> rhythm guitars -> harmonic support -> lead guitars -> lead vocal -> backing vocals -> FX. The order is a policy variable, not a universal truth. Preserve intermediate submixes so later decisions are inspectable and reversible.

This repository currently implements the orchestration principle, not the paper authors' latent flow-matching model.

## DiffVox

Sources:
- https://arxiv.org/abs/2504.14735
- https://github.com/SonyResearch/diffvox

Treat vocal EQ, dynamics, delay and reverb parameters as a coupled space. Prefer coupled preset/prior or inference-time optimisation proposals over completely independent knob optimisation when style matching a vocal.

The official SonyResearch repository is an external offline backend. Any parameters produced by it are proposals and must pass A/B checks before being promoted.

## Safety / evaluation contract

- Never mark these research rules `auto_apply` before audio validation.
- Never claim the research model is deployed unless its runtime status confirms the actual code/checkpoints are available.
- Protect lead-vocal intelligibility, kick/bass anchor, section dynamics, transients and mono compatibility.
- Reject candidates when objective metrics improve but listening preference or musical hierarchy degrades.
- Log source IDs, before/after metrics and human feedback for every promoted decision.
