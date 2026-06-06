# Proposal: operator-proposal-apply-sequence-execution-audit-20260606

## Short Thesis

Add `audit_operator_proposal_apply_sequence_execution` as the read-only proof
layer after sequence execution.

## What I Understood

The new executor moves the system closer to Supervised AutoMix, but autonomous
systems need post-execution evidence, not just an apply result. The audit should
prove the executor stayed inside the approved fader/gain live boundary and
delegated writes to the existing apply handler.

## Proposed Solution

- Accept `execution_result_payload` or direct execution rows.
- Validate summary and safety envelope:
  - `uses_existing_apply_operator_proposal=true`;
  - `live_apply_entrypoint=apply_operator_proposal`;
  - no direct live write or direct manual console write claim;
  - no executor-direct idempotency reservation;
  - no live widening.
- Validate applied rows:
  - row status is `applied`;
  - apply request is `apply_operator_proposal` with approved idempotency;
  - apply result status is `applied`;
  - value type is `fader` or `gain`;
  - fader effective value is not above `0 dBFS`;
  - readback/proof evidence passes the existing verified manual result helper.
- Emit audit rows and a read-only summary.
- Advertise the audit handler from Decision Queue apply-sequence visibility.

## Likely Files To Touch

- `backend/handlers/product_state_handlers.py`
- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-execution-audit-20260606.md`

## Alternatives Considered

- Store audit records in durable runtime memory now. Rejected because that adds
  a write surface; this slice should first define and test the proof contract.
- Treat executor `status=applied` as sufficient. Rejected because forged summary
  flags or missing row evidence would be indistinguishable from safe execution.
- Re-run apply preflight during audit. Rejected because audit should be
  read-only and should not depend on current queue state after execution.

## Risks

The audit can be mistaken for future execution permission. Mitigation: every
payload carries `audit_only=true`, `execution_allowed_by_audit=false`, and
`live_authority_granted=false`.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_audit_operator_proposal_apply_sequence_execution_passes_verified_executor_result -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_audit_operator_proposal_apply_sequence_execution_blocks_forged_direct_write_claim -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Where The Other Agent May Disagree

Kimi may prefer persisting audit records immediately. I would keep persistence
separate so the first audit contract remains purely read-only and easy to
review.
