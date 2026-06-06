# ADR: operator-proposal-apply-sequence-executor-20260606

## Context

Apply-sequence planning, dry-run rehearsal, and execution gating now exist, but
the system still cannot execute a gated sequence. The project goal requires
movement toward supervised and eventually autonomous execution without creating
unsafe live-write paths.

## Options Considered

1. Add a sequence executor that calls `_apply_manual_console_write` directly.
2. Add a sequence executor that delegates to `apply_operator_proposal`.
3. Keep execution out of scope and stop at gate output.

## Decision

Use option 2. Add `execute_operator_proposal_apply_sequence` as a supervised
executor that validates gate-ready rows and delegates every apply to the
existing `apply_operator_proposal` handler.

## Why This Won

It creates real sequence execution progress while preserving the existing
single live-write boundary. The executor can order and summarize work, but the
actual idempotency reservation, safety validation, queue mutation, and manual
console write remain inside the current apply path.

## Rejected Alternatives

- Direct manual console writes were rejected because they would bypass the
  established supervised apply gate.
- Stopping at gate output was rejected because the next practical step toward
  supervised automation is an executor, not another read-only marker.
- Skipping invalid rows was rejected because mixed-validity execution can hide
  stale or forged input. The first executor fails closed before attempting any
  row if validation finds a blocker.

## Implementation Plan

1. Add executor request parsing for `execution_gate_payload` and direct
   `ready_for_executor` rows.
2. Require `execute=true`, `operator_ack=true`, `approved=true`, and supervised
   apply capability.
3. Validate all candidate rows against current queue state, apply request,
   idempotency, and fader/gain-only normalized changes.
4. Delegate ready rows to `handle_apply_operator_proposal`.
5. Emit a final sequence execution summary after per-proposal apply messages.
6. Add Decision Queue executor discovery fields.
7. Update durable project memory with the live-boundary rule.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_execute_operator_proposal_apply_sequence_applies_gate_ready_row_via_apply_handler -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_execute_operator_proposal_apply_sequence_requires_execute_ack_and_approval -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_execute_operator_proposal_apply_sequence_blocks_stale_gate_row_before_write -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks And Rollback

Risk: sequence execution expands perceived live authority. Mitigation: the
executor routes only through `apply_operator_proposal`, preserves fader/gain as
the only allowed live write kinds, and fails closed on stale or forged gate
rows.

Rollback: remove the executor handler and Decision Queue discovery fields.
Plan, dry-run, gate, and single-proposal apply remain available.
