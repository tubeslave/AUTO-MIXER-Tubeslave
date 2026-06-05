# Proposal: compressor-proposal-freshness-gate-20260605

## Short thesis

Use current compressor-state cache freshness as a proposal-only gate before
showing compressor dynamics moves.

## What I understood

The project needs autonomous compression, but live compressor writes are not
approved. The useful step now is to make compressor recommendations more
truthful: if the current compressor state is stale or incomplete, the system
should not present a confident compressor move as proposal-only.

## Proposed solution

Extend `CompressorAnalysisInput` with optional
`current_state_cache_freshness`. Normalize that payload into evidence. If it is
provided and the status is not `fresh`, add
`compressor_current_state_cache_not_fresh` and demote the recommendation to
`analysis_only`. Keep the module transport-free and keep operator proposals
without `requested_change`.

## Likely files to touch

- `backend/compressor_dynamics_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `tests/test_compressor_dynamics_proposal.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Require freshness for every compressor proposal immediately. Rejected to avoid
  breaking legacy audio-only analysis until all callers pass product snapshot
  freshness.
- Ignore freshness until physical query fixtures exist. Rejected because cached
  field freshness is already available and can prevent misleading proposals.
- Promote compressor writes. Rejected because dynamics readback, rollback, and
  physical proof are still incomplete.

## Risks

- Callers that start providing freshness may see fewer compressor proposals when
  cache timestamps are stale.
- Missing freshness remains a legacy gap until upstream integration is complete.

## Test plan

- Focused compressor proposal tests.
- Proposal registry/adapter tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue missing freshness should block all compressor proposals. I agree
for promotion, but this cycle keeps legacy analysis compatible while blocking
known-bad stale evidence.
