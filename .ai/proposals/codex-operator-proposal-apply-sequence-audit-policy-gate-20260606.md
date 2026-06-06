# Proposal: operator-proposal-apply-sequence-audit-policy-gate-20260606

## Short Thesis

Add a read-only policy gate after sequence execution audit so future Supervised
AutoMix policy consumes only verified, non-authoritative evidence.

## What I Understood

The project needs to move toward autonomous execution, but future policy
consumers must not treat raw executor output or even raw audit output as live
authority. A small gate can expose passed audit rows as policy evidence while
keeping execution and persistence separate.

## Proposed Solution

- Add `gate_operator_proposal_apply_sequence_audit_policy`.
- Accept `execution_audit_payload` or direct `audit_rows`.
- Require audit payload type `operator_proposal_apply_sequence_execution_audit`
  and `status=passed`.
- Validate summary and safety envelope:
  - `audit_only=true`;
  - `execution_allowed_by_audit=false`;
  - no live write, queue mutation, idempotency reservation, manual write, live
    authority, or live widening;
  - `allowed_live_write_kinds` is exactly within fader/gain.
- Validate rows:
  - status is `passed`;
  - value type is fader/gain;
  - idempotency and verified readback/proof evidence are present.
- Emit `policy_evidence_ready` candidates and read-only safety flags.
- Advertise the policy gate from Decision Queue apply-sequence visibility.

## Likely Files To Touch

- `backend/handlers/product_state_handlers.py`
- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-audit-policy-gate-20260606.md`

## Alternatives Considered

- Let future policy consume audit rows directly. Rejected because the gate
  gives policy consumers a stronger, explicit contract.
- Persist policy evidence now. Rejected because persistence needs a separate
  operator-acknowledged write contract.
- Let the gate trigger the next sequence automatically. Rejected because this
  slice is policy evidence only, not execution authority.

## Risks

Policy evidence may be mistaken for permission to apply. Mitigation: payloads
include `execution_allowed_by_policy_gate=false`, `policy_mutation=false`,
`live_authority_granted=false`, and `full_autonomous_ready=false`.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_gate_operator_proposal_apply_sequence_audit_policy_marks_passed_audit_ready -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_gate_operator_proposal_apply_sequence_audit_policy_blocks_forged_live_authority -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Where The Other Agent May Disagree

Kimi may prefer jumping straight to persistent policy learning. I would keep
this as a non-mutating gate first because it defines the evidence contract that
persistence can later consume.
