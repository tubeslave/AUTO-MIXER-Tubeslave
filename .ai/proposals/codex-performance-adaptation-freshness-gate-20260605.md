# Proposal: performance-adaptation-freshness-gate-20260605

## Short thesis

Use current performance context freshness as a proposal-only gate before showing
live drift correction recommendations.

## What I understood

The Automixer should eventually react during a concert, but live adaptation is
safety-sensitive. A stale level trend or current readback can produce a plausible
but wrong ride recommendation. The next useful step is to consume cached
freshness evidence while keeping the module transport-free and non-queueable.

## Proposed solution

Extend `PerformanceAdaptationAnalysisInput` with optional
`current_state_cache_freshness`. Normalize that payload into recommendation
evidence. If it is provided and the status is not `fresh`, add
`performance_adaptation_current_state_cache_not_fresh` and demote the
recommendation to `analysis_only`. Keep operator proposals without
`requested_change` or `requested_changes`.

## Likely files to touch

- `backend/performance_adaptation_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_performance_adaptation_proposal.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Require freshness for every performance adaptation proposal immediately.
  Rejected to avoid breaking legacy analysis before all callers pass product
  snapshot freshness.
- Ignore freshness until live replay fixtures exist. Rejected because cached
  freshness already identifies known-stale trend/readback context.
- Add direct fader/gain apply from performance adaptation. Rejected because
  replay match, rate limiting, rollback proof, and hardware proof are not
  complete for this action family.

## Risks

- Callers that provide stale freshness will produce fewer adaptation proposals.
- Missing freshness remains a legacy gap until upstream integration is complete.

## Test plan

- Focused performance adaptation proposal tests.
- Registry/capability contract tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue missing freshness should block all live adaptation proposals. I
agree for promotion, but this cycle keeps legacy analysis compatible while
blocking known stale evidence.
