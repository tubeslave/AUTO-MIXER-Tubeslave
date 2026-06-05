# Proposal: decision-evidence-packet-freshness-gate-20260605

## Short thesis

Use current evidence-context freshness as a proposal-only gate before showing
decision evidence packets.

## What I understood

The Automixer should explain every decision with audio, psychoacoustic,
statistical, history, reference, inter-channel, arrangement, genre, and
previous-outcome evidence. A stale evidence packet can look complete while being
wrong. The next useful step is to consume cached freshness evidence while
keeping the module transport-free and non-queueable.

## Proposed solution

Extend `DecisionEvidenceAnalysisInput` with optional
`current_state_cache_freshness`. Normalize that payload into recommendation
evidence. If it is provided and the status is not `fresh`, add
`decision_evidence_context_cache_not_fresh` and demote the recommendation to
`analysis_only`. Keep operator proposals without `requested_change` or
`requested_changes`.

## Likely files to touch

- `backend/decision_evidence_packet_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_decision_evidence_packet_proposal.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Require freshness for every decision evidence packet immediately. Rejected to
  avoid breaking legacy analysis before all callers pass product snapshot
  freshness.
- Ignore freshness until replay learning consumes packets. Rejected because
  stale context can already be identified and blocked at the packet boundary.
- Allow evidence packets to queue/apply actions. Rejected because packets are
  review contracts, not live authority.

## Risks

- Callers that provide stale freshness will produce fewer evidence packets.
- Missing freshness remains a legacy gap until upstream integration is complete.

## Test plan

- Focused decision evidence packet tests.
- Registry/capability contract tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue missing freshness should block all decision packets. I agree for
promotion, but this cycle keeps legacy analysis compatible while blocking known
stale evidence.
