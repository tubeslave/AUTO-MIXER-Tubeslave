# ADR: operator-proposal-apply-sequence-policy-memory-draft-20260606

## Context

The sequence audit policy gate can expose verified supervised fader/gain
execution as policy evidence. The long-term Automixer goal requires learning
from previous decisions, but memory writes are a separate authority boundary.

## Options Considered

1. Persist policy evidence immediately.
2. Add a read-only memory-draft handler.
3. Let future memory tooling consume policy gate payloads directly.

## Decision

Use option 2. Add `draft_operator_proposal_apply_sequence_policy_memory` to turn
policy evidence into deterministic memory draft records without writing memory.

## Why This Won

It advances the learning loop while preserving safe authority boundaries. Future
append-only memory writers can consume a clear draft schema instead of parsing
raw policy-gate payloads.

## Rejected Alternatives

- Immediate persistence was rejected because it needs explicit operator
  acknowledgement and append-only writer review.
- Direct policy-gate consumption was rejected because it would couple future
  memory tooling to a policy eligibility payload.
- Combining policy gate and draft was rejected because readiness for policy and
  readiness for memory review are different contracts.

## Implementation Plan

1. Add memory-draft request parsing for `policy_gate_payload` and direct
   `policy_evidence`.
2. Validate policy-gate summary and safety flags.
3. Validate fader/gain-only policy evidence rows.
4. Emit no-write memory draft records and blockers.
5. Advertise the draft handler in Decision Queue apply-sequence visibility.
6. Update durable memory guidance.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_draft_operator_proposal_apply_sequence_policy_memory_creates_read_only_draft -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_draft_operator_proposal_apply_sequence_policy_memory_blocks_forged_memory_write_claim -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks And Rollback

Risk: a draft can be mistaken for a memory write. Mitigation: the draft payload
sets `append_only_memory_write_allowed_by_draft=false`,
`requires_separate_memory_writer=true`, `memory_mutation=false`, and
`live_authority_granted=false`.

Rollback: remove the draft handler and Decision Queue discovery fields. Policy
gate, audit, executor, and earlier sequence stages remain available.
