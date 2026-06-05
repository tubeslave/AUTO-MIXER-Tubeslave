# Brief: decision-evidence-packet-freshness-gate-20260605

## Context

The project goal requires every decision to be based on audio analysis,
psychoacoustics, statistical models, history, references, previous outcomes,
inter-channel analysis, arrangement, and genre. The current implementation
exposes this as proposal-only decision evidence packets. A packet can satisfy
family coverage and confidence thresholds while still being unsafe if the
underlying evidence snapshot is stale.

## Task

Add a read-only freshness gate to `decision_evidence_packet_proposal`: when
current decision-evidence context freshness is supplied and is not fresh, demote
the recommendation to `analysis_only` with an explicit blocker. Fresh context
should remain eligible for proposal-only output.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not add action apply support from evidence packets.
- Keep decision evidence output proposal-only and non-queueable.
- Do not query a console from the proposal module.

## Expected Evidence

- Fresh current-state freshness is recorded in decision evidence packet output.
- Stale current-state freshness blocks proposal-only output.
- `build_decision_evidence_packet_proposals` filters stale-context items.
- Proposal-only registry/contract metadata mentions current decision-evidence
  context freshness.
