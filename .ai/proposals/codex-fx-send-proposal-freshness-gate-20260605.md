# Proposal: fx-send-proposal-freshness-gate-20260605

## Short thesis

Use current FX send/return cache freshness as a proposal-only gate before showing
FX send moves.

## What I understood

The Automixer should eventually manage FX and effect sends automatically, but it
cannot safely promote FX decisions while the current send, return, or FX-slot
state is stale or unknown. The next useful step is to consume cached readback
freshness as evidence, not to add live writes.

## Proposed solution

Extend `FXSendAnalysisInput` with optional `current_state_cache_freshness`.
Normalize that payload into recommendation evidence. If it is provided and the
status is not `fresh`, add `fx_current_state_cache_not_fresh` and demote the
recommendation to `analysis_only`. Keep the module transport-free and keep
operator proposals without `requested_change`.

## Likely files to touch

- `backend/fx_send_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `tests/test_fx_send_proposal.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Require freshness for every FX proposal immediately. Rejected to avoid
  breaking legacy audio-only analysis before all callers pass product snapshot
  freshness.
- Ignore freshness until physical FX query fixtures exist. Rejected because
  cached freshness is already available and can block known-bad state.
- Add FX live writes. Rejected because send/return readback, feedback checks,
  rollback, and physical proof are incomplete.

## Risks

- Callers that provide stale freshness will produce fewer FX proposals.
- Missing freshness remains a legacy gap until upstream integration is complete.

## Test plan

- Focused FX proposal tests.
- Registry/capability contract tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue missing freshness should block all FX proposals. I agree for
promotion, but this cycle keeps legacy analysis compatible while blocking known
stale evidence.
