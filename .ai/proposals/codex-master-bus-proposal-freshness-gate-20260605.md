# Proposal: master-bus-proposal-freshness-gate-20260605

## Short thesis

Use current main-bus processing cache freshness as a proposal-only gate before
showing master-bus optimization moves.

## What I understood

The Automixer should eventually optimize the master bus automatically, but it
cannot safely promote master-bus decisions while the current limiter, EQ,
dynamics, or matrix-related state is stale or unknown. The next useful step is
to consume cached readback freshness as evidence, not to add live writes.

## Proposed solution

Extend `MasterBusAnalysisInput` with optional `current_state_cache_freshness`.
Normalize that payload into recommendation evidence. If it is provided and the
status is not `fresh`, add `master_bus_current_state_cache_not_fresh` and demote
the recommendation to `analysis_only`. Keep the module transport-free and keep
operator proposals without `requested_change` or `requested_changes`.

## Likely files to touch

- `backend/master_bus_optimization_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `tests/test_master_bus_optimization_proposal.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Require freshness for every master-bus proposal immediately. Rejected to avoid
  breaking legacy audio-only analysis before all callers pass product snapshot
  freshness.
- Ignore freshness until physical main-bus query fixtures exist. Rejected
  because cached freshness is already available and can block known-bad state.
- Add master-bus live writes. Rejected because main-bus readback, limiter and
  dynamics proof, rollback, and physical proof are incomplete.

## Risks

- Callers that provide stale freshness will produce fewer master-bus proposals.
- Missing freshness remains a legacy gap until upstream integration is complete.

## Test plan

- Focused master-bus proposal tests.
- Registry/capability contract tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue missing freshness should block all master-bus proposals. I agree
for promotion, but this cycle keeps legacy analysis compatible while blocking
known stale evidence.
