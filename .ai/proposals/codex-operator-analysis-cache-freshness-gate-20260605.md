# Proposal: operator-analysis-cache-freshness-gate-20260605

## Short thesis

Use cache freshness as a promotion gate for operator-analysis proposals before
they become applyable queue items.

## What I understood

The goal requires autonomous decisions to be grounded in current console state.
Operator analysis is already a producer of fader/gain proposals, so it is the
right first place to enforce the freshness contract created by the read DTO
work.

## Proposed solution

Add read-only freshness checks inside `backend/operator_analysis.py` for
requested fader/gain changes. A fresh timestamp adds safety evidence to the
requested change. Stale, unknown, missing, or unsupported cache freshness keeps
the observation and queue item, but strips `requested_change` or
`requested_changes`, marks the proposal `analysis`, and stores blocker metadata
in raw payload fields.

## Likely files to touch

- `backend/operator_analysis.py`
- `backend/operator_proposal_queue.py`
- `tests/test_operator_product_state.py`

## Alternatives considered

- Query the console before creating proposals. Rejected because operator
  analysis must remain read-only and transport-free.
- Keep proposals applyable and rely only on supervised pre-read. Rejected
  because stale cached values should not pass the earlier promotion gate.
- Drop stale observations entirely. Rejected because the operator should still
  see the audio issue.

## Risks

- Queue may contain more non-applyable analysis items when cache timestamps are
  missing.
- This is cache freshness only; physical readback proof is still required later
  in the supervised apply path.

## Test plan

- Focused operator Product Layer tests for fresh and stale cache paths.
- Broad server/readiness/operator policy tests.
- Full pytest suite and diff check.

## Where the other agent may disagree

Kimi may prefer blocking proposal creation entirely. My counterpoint is that
keeping the observation visible improves explainability while still preventing
unsafe apply.
