# Proposal: pan-proposal-freshness-gate-20260605

## Short thesis

Use current pan cache freshness as a proposal-only gate before showing pan moves.

## What I understood

The Automixer should eventually pan channels automatically, but it cannot safely
promote pan decisions while the current pan state is stale or unknown. The next
useful step is to consume cached readback freshness as evidence, not to add live
writes.

## Proposed solution

Extend `PanAnalysisInput` with optional `current_state_cache_freshness`.
Normalize that payload into recommendation evidence. If it is provided and the
status is not `fresh`, add `pan_current_state_cache_not_fresh` and demote the
recommendation to `analysis_only`. Keep the module transport-free and keep
operator proposals without `requested_change`.

## Likely files to touch

- `backend/pan_spatial_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `tests/test_pan_spatial_proposal.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Require freshness for every pan proposal immediately. Rejected to avoid
  breaking legacy audio-only analysis before all callers pass product snapshot
  freshness.
- Ignore freshness until physical pan query fixtures exist. Rejected because
  cached freshness is already available and can block known-bad state.
- Add pan live writes. Rejected because readback, rollback, mono compatibility,
  and physical proof are incomplete.

## Risks

- Callers that provide stale freshness will produce fewer pan proposals.
- Missing freshness remains a legacy gap until upstream integration is complete.

## Test plan

- Focused pan proposal tests.
- Registry/capability contract tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue missing freshness should block all pan proposals. I agree for
promotion, but this cycle keeps legacy analysis compatible while blocking known
stale evidence.
