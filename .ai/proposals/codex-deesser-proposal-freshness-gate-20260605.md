# Proposal: deesser-proposal-freshness-gate-20260605

## Short thesis

Use current dynamics/dynamic-EQ cache freshness as a proposal-only gate before
showing de-esser moves.

## What I understood

The Automixer should eventually de-ess automatically, but it cannot safely
promote de-esser decisions while the current processing state is stale or
unknown. The next useful step is to consume cached readback freshness as
evidence, not to add live writes.

## Proposed solution

Extend `DeesserAnalysisInput` with optional `current_state_cache_freshness`.
Normalize that payload into recommendation evidence. If it is provided and the
status is not `fresh`, add `deesser_current_state_cache_not_fresh` and demote
the recommendation to `analysis_only`. Keep the module transport-free and keep
operator proposals without `requested_change`.

## Likely files to touch

- `backend/deesser_vocal_sibilance.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `tests/test_deesser_vocal_sibilance.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Require freshness for every de-esser proposal immediately. Rejected to avoid
  breaking legacy audio-only analysis before all callers pass product snapshot
  freshness.
- Ignore freshness until physical query fixtures exist. Rejected because cached
  freshness is already available and can block known-bad state.
- Add de-esser live writes. Rejected because readback, rollback, and physical
  proof are incomplete.

## Risks

- Callers that provide stale freshness will produce fewer de-esser proposals.
- Missing freshness remains a legacy gap until upstream integration is complete.

## Test plan

- Focused de-esser proposal tests.
- Registry/capability contract tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue missing freshness should block all de-esser proposals. I agree
for promotion, but this cycle keeps legacy analysis compatible while blocking
known stale evidence.
