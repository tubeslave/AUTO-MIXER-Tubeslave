# Proposal: bus-group-balance-freshness-gate-20260605

## Short thesis

Use current bus/group/DCA/matrix cache freshness as a proposal-only gate before
showing grouped balance moves.

## What I understood

The Automixer should eventually balance stems, groups, DCAs, buses, and matrices
automatically, but it cannot safely promote those decisions while assignment,
fader, or matrix state is stale or unknown. The next useful step is to consume
cached readback freshness as evidence, not to add live writes.

## Proposed solution

Extend `BusGroupBalanceAnalysisInput` with optional
`current_state_cache_freshness`. Normalize that payload into recommendation
evidence. If it is provided and the status is not `fresh`, add
`bus_group_current_state_cache_not_fresh` and demote the recommendation to
`analysis_only`. Keep the module transport-free and keep operator proposals
without `requested_change` or `requested_changes`.

## Likely files to touch

- `backend/bus_group_balance_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_bus_group_balance_proposal.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Require freshness for every bus/group balance proposal immediately. Rejected
  to avoid breaking legacy audio-only analysis before all callers pass product
  snapshot freshness.
- Ignore freshness until physical assignment query fixtures exist. Rejected
  because cached freshness is already available and can block known-bad state.
- Add bus/group/DCA/matrix live writes. Rejected because assignment readback,
  true-peak safety, rollback, and physical proof are incomplete.

## Risks

- Callers that provide stale freshness will produce fewer bus/group balance
  proposals.
- Missing freshness remains a legacy gap until upstream integration is complete.

## Test plan

- Focused bus/group balance proposal tests.
- Registry/capability contract tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue missing freshness should block all bus/group balance proposals. I
agree for promotion, but this cycle keeps legacy analysis compatible while
blocking known stale evidence.
