# Proposal: operator-proposal-apply-sequence-executor-20260606

## Short Thesis

Add the first supervised apply-sequence executor, but make it a thin,
auditable wrapper around `apply_operator_proposal` rather than a new console
write path.

## What I Understood

The long-term goal requires autonomous execution, but this repository currently
allows live writes only through supervised fader/gain proposal apply. The new
executor should advance automation by sequencing already gated proposals while
preserving that live boundary.

## Proposed Solution

- Add `execute_operator_proposal_apply_sequence`.
- Accept `execution_gate_payload` or direct `ready_for_executor` rows.
- Require explicit `execute=true`, `operator_ack=true`, and `approved=true`.
- Require supervised operator mode capability `can_apply_to_console`.
- Validate every candidate before applying any row:
  - row status is `ready_for_executor`;
  - gate row does not claim direct execution authority;
  - current queue proposal is still pending, applyable, unexpired, and
    unchanged from the gate snapshot;
  - `apply_request.type` is `apply_operator_proposal`;
  - idempotency key is present, not already reserved, and unique in sequence;
  - normalized changes are fader/gain only and respect the fader ceiling.
- Delegate each row to `handle_apply_operator_proposal` and emit a final
  sequence summary after the per-proposal apply messages.
- Stop on the first delegated apply that is not applied.
- Advertise the executor from Decision Queue apply-sequence visibility.

## Likely Files To Touch

- `backend/handlers/product_state_handlers.py`
- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-executor-20260606.md`

## Alternatives Considered

- Call `_apply_manual_console_write` directly. Rejected because it would create
  a second live-write path.
- Let the executor consume dry-run rows directly. Rejected because the existing
  execution gate is the stricter precondition contract.
- Apply valid rows and skip invalid rows. Rejected for now; all candidates must
  validate before any live apply is attempted.

## Risks

The executor can be mistaken as authority to widen live writes. Mitigation:
payloads name `uses_existing_apply_operator_proposal=true`,
`direct_manual_console_write_called=false`, `allowed_live_write_kinds=["fader",
"gain"]`, and `requires_gate_ready=true`.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_execute_operator_proposal_apply_sequence_applies_gate_ready_row_via_apply_handler -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_execute_operator_proposal_apply_sequence_requires_execute_ack_and_approval -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_execute_operator_proposal_apply_sequence_blocks_stale_gate_row_before_write -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Where The Other Agent May Disagree

Kimi may prefer keeping the executor read-only for another cycle. I would land
this limited executor because it adds real supervised progress while using the
existing single-proposal live gate unchanged.
