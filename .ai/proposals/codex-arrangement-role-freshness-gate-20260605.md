# Proposal: arrangement-role-freshness-gate-20260605

## Short thesis

Use current arrangement-role context freshness as a proposal-only gate before
showing source role recommendations.

## What I understood

The Automixer should infer source roles in the arrangement, but those roles
shape many downstream mix decisions. A stale section, inter-channel, reference,
or accepted-history context can make a plausible role recommendation wrong. The
next useful step is to consume cached freshness evidence while keeping the
module transport-free and non-queueable.

## Proposed solution

Extend `ArrangementRoleAnalysisInput` with optional
`current_state_cache_freshness`. Normalize that payload into recommendation
evidence. If it is provided and the status is not `fresh`, add
`arrangement_role_context_cache_not_fresh` and demote the recommendation to
`analysis_only`. Keep operator proposals without `requested_change` or
`requested_changes`.

## Likely files to touch

- `backend/arrangement_role_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_arrangement_role_proposal.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Require freshness for every arrangement-role proposal immediately. Rejected
  to avoid breaking legacy analysis before all callers pass product snapshot
  freshness.
- Ignore freshness until role-learning replay exists. Rejected because stale
  context can already be identified and blocked at the proposal boundary.
- Add downstream role apply or learning authority. Rejected because accepted
  operator decisions and replay correlation are not complete.

## Risks

- Callers that provide stale freshness will produce fewer arrangement-role
  proposals.
- Missing freshness remains a legacy gap until upstream integration is complete.

## Test plan

- Focused arrangement-role proposal tests.
- Registry/capability contract tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue missing freshness should block all arrangement-role proposals. I
agree for promotion, but this cycle keeps legacy analysis compatible while
blocking known stale evidence.
