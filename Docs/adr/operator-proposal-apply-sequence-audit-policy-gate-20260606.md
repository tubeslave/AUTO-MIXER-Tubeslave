# ADR: operator-proposal-apply-sequence-audit-policy-gate-20260606

## Context

Sequence execution audit can prove that supervised fader/gain sequence execution
used the approved path and produced verified evidence. Future automation policy
needs a stable input contract over that audit output before it can safely learn
from or react to execution outcomes.

## Options Considered

1. Let policy consumers read audit payloads directly.
2. Add a read-only policy gate over passed audit evidence.
3. Persist policy evidence immediately.

## Decision

Use option 2. Add `gate_operator_proposal_apply_sequence_audit_policy` as a
read-only gate that turns passed audit rows into policy evidence candidates
without granting execution, persistence, or live authority.

## Why This Won

It creates a policy-consumable contract while preserving the existing execution
and safety boundaries. Future policy/persistence work can depend on explicit
`policy_evidence_ready` records instead of parsing raw executor or audit
payloads.

## Rejected Alternatives

- Direct audit consumption was rejected because audit payloads are still a proof
  report, not a policy contract.
- Immediate persistence was rejected because it adds a write surface and needs a
  separate acknowledged memory/policy-store contract.
- Automatic follow-up execution was rejected because passed audit evidence is
  not permission to perform more live writes.

## Implementation Plan

1. Add policy-gate request parsing for `execution_audit_payload` or direct rows.
2. Validate audit summary and safety envelope.
3. Validate passed fader/gain audit rows.
4. Emit read-only policy evidence rows and blockers.
5. Advertise the gate in Decision Queue apply-sequence visibility.
6. Add durable memory guidance.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_gate_operator_proposal_apply_sequence_audit_policy_marks_passed_audit_ready -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_gate_operator_proposal_apply_sequence_audit_policy_blocks_forged_live_authority -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks And Rollback

Risk: policy evidence may be confused with live execution authority. Mitigation:
the gate sets `execution_allowed_by_policy_gate=false`,
`live_authority_granted=false`, `policy_mutation=false`, and
`full_autonomous_ready=false`.

Rollback: remove the policy gate handler and Decision Queue discovery fields.
Execution, audit, and earlier sequence stages remain available.
