# Proposal: operator-proposal-apply-preflight-20260606

## Short Thesis

Add a dry-run `preflight_operator_proposal_apply` Product Layer handler that
answers "would this pending proposal pass the supervised apply gate now?"
without performing the apply.

## What I understood

The project is moving toward more autonomous mixing through proposals and
operator queues. Before widening execution, the operator and UI need a reliable
preflight result that uses the same safety checks as apply while preserving the
current live-write boundary.

## Proposed solution

- Add `preflight_operator_proposal_apply` next to `apply_operator_proposal` in
  `backend/handlers/product_state_handlers.py`.
- Reuse existing local helpers for proposal lookup, linked-change
  normalization, pre-read validation, delta limits, upward move checks, and
  cumulative delta limits.
- Add a no-mutation flag to cumulative-delta validation so preflight cannot
  rewrite delta history.
- Return a compact result with `ready_to_apply`, `reason`, normalized changes,
  idempotency preflight, operator mode, and no-live-write safety flags.
- Register the handler in the Product Layer dispatch map.

## Likely files to touch

- `backend/handlers/product_state_handlers.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-preflight-20260606.md`

## Alternatives considered

- Let UI call `apply_operator_proposal` with `approved=false`. Rejected because
  it can still reserve idempotency keys before later blockers.
- Duplicate validation in a separate module. Rejected for this small slice
  because the apply validation helpers are currently local to handler
  registration.
- Widen live execution for source-identity imported proposals. Rejected because
  source identity queue import is provenance, not physical readback or rollback
  proof.

## Risks

Preflight could drift from apply if validation is duplicated too aggressively.
Mitigation: reuse the same local validation helper and only add a no-mutation
mode for cumulative delta checks.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_preflight_operator_proposal_apply_reports_ready_without_live_write tests/test_operator_product_state.py::test_preflight_operator_proposal_apply_blocks_stale_pre_read_without_mutation tests/test_operator_product_state.py::test_preflight_operator_proposal_apply_blocks_unsupported_kind_without_mutation -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Where the other agent may disagree

Kimi may prefer extracting apply validation into a shared service first. I would
land the handler-local preflight now because it gives immediate operator
visibility and can later be extracted with tests preserving the contract.
