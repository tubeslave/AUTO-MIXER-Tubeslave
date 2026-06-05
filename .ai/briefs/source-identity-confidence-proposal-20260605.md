# Brief: source-identity-confidence-proposal-20260605

## Context

The project goal requires reliable automatic source type classification before
downstream EQ, dynamics, FX, arrangement-role, and frequency-separation proposals
can trust source-specific assumptions. The capability matrix already tracks
source type classification as canonical analysis, but the confidence gap was
still only a next-step note.

The worktree is dirty from earlier proposal-only safety and freshness cycles.
This task must not revert unrelated local changes.

## Task

Add a transport-free `source_identity_confidence_proposal` layer. It should turn
unknown, low-confidence, ambiguous, or unreliable classifier results into
operator-facing source identity review records. It should also expose downstream
rejection reasons and priority metadata so later proposal families can hold or
de-prioritize source-specific decisions until labels are accepted and replayed.

## Safety constraints

- Do not add live OSC, WING, dLive, or Mixing Station writes.
- Do not create queueable live actions.
- Do not run ML inference, query hardware, or read audio from this review layer.
- Block proposals when provided source-identity context freshness is stale.
- Keep downstream processing changes outside this module.

## Acceptance criteria

- Fresh context can emit source-identity review proposals for low-confidence or
  ambiguous classifier outputs.
- Stale provided context blocks source-identity review proposals.
- Proposal records include classifier source, match type, confidence,
  alternative margin, operator-acceptance requirement, and downstream rejection
  reasons.
- Registry, proposal-only contract, and capability matrix reference the new
  review layer without widening live writes.
- Focused tests and the standard test command pass.
