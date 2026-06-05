# Proposal: routing-management-freshness-gate-20260605

## Short thesis

Use current route cache freshness as a proposal-only gate before showing routing
diff recommendations.

## What I understood

The Automixer should eventually control routing automatically, but route changes
can rewrite broad mixer state. It cannot safely promote routing decisions while
current source, proposed source, rollback, or snapshot state is stale or
unknown. The next useful step is to consume cached readback freshness as
evidence, not to add live writes.

## Proposed solution

Extend `RoutingManagementAnalysisInput` with optional
`current_state_cache_freshness`. Normalize that payload into recommendation
evidence. If it is provided and the status is not `fresh`, add
`routing_current_state_cache_not_fresh` and demote the recommendation to
`analysis_only`. Keep the module transport-free and keep operator proposals
without `requested_change` or `requested_changes`.

## Likely files to touch

- `backend/routing_management_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_routing_management_proposal.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Require freshness for every routing proposal immediately. Rejected to avoid
  breaking legacy analysis before all callers pass product snapshot freshness.
- Ignore freshness until physical route query fixtures exist. Rejected because
  cached freshness is already available and can block known-bad state.
- Add routing live writes. Rejected because full route readback, rollback,
  scope limits, operator confirmation, and physical proof are incomplete.

## Risks

- Callers that provide stale freshness will produce fewer routing proposals.
- Missing freshness remains a legacy gap until upstream integration is complete.

## Test plan

- Focused routing management proposal tests.
- Registry/capability contract tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue missing freshness should block all routing proposals. I agree for
promotion, but this cycle keeps legacy analysis compatible while blocking known
stale evidence.
