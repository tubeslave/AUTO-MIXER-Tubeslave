# Proposal: operator-proposal-apply-sequence-policy-memory-draft-20260606

## Short Thesis

Add a read-only memory-draft stage after sequence audit policy gating so
verified supervised execution can become learning material without writing
memory yet.

## What I Understood

The project goal explicitly requires long-term memory and improvement from
previous decisions. The current pipeline can prove a supervised fader/gain
sequence was safely executed and policy-gated. The next useful step is to draft
what would be learned, while keeping persistence separate and explicit.

## Proposed Solution

- Add `draft_operator_proposal_apply_sequence_policy_memory`.
- Accept `policy_gate_payload` or direct `policy_evidence`.
- Require policy-gate payload type
  `operator_proposal_apply_sequence_audit_policy_gate` and `status=ready`.
- Reject policy-gate payloads that claim execution permission, live authority,
  policy mutation, runtime memory mutation, model mutation, live widening, or
  Full Autonomous readiness.
- Validate each policy evidence row is `policy_evidence_ready`, fader/gain only,
  has idempotency, verified readback/proof, rollback, and
  `live_apply_entrypoint=apply_operator_proposal`.
- Emit deterministic draft records with no write permission.
- Advertise the draft handler from Decision Queue apply-sequence visibility.

## Likely Files To Touch

- `backend/handlers/product_state_handlers.py`
- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-draft-20260606.md`

## Alternatives Considered

- Write the memory immediately. Rejected because persistence needs a separate
  append-only writer with explicit operator acknowledgement.
- Let future memory tooling consume policy gate rows directly. Rejected because
  a draft schema makes the learning contract explicit.
- Fold the draft into the policy gate. Rejected because policy eligibility and
  memory drafting are different review steps.

## Risks

Draft records may be mistaken for durable memory. Mitigation: every payload and
record carries `append_only_memory_write_allowed_by_draft=false`,
`memory_mutation=false`, and `requires_separate_memory_writer=true`.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_draft_operator_proposal_apply_sequence_policy_memory_creates_read_only_draft -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_draft_operator_proposal_apply_sequence_policy_memory_blocks_forged_memory_write_claim -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Where The Other Agent May Disagree

Kimi may prefer writing accepted policy evidence immediately. I would keep this
as a draft first because it gives the future writer a small, auditable input
contract.
