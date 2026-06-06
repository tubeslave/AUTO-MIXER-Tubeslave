# Proposal: decision-queue-batch-preflight-visibility-20260606

## Short thesis

Add a read-only `batch_preflight` summary to the decision queue so the cockpit
can discover the explicit batch dry-run path before any supervised apply.

## What I understood

Single-proposal preflight and batch preflight are handler actions. The decision
queue snapshot should not run them, but it should make the next safe operator
step visible.

## Proposed solution

- Add `_decision_queue_batch_preflight_summary` in
  `backend/operator_product_state.py`.
- Derive default batch candidates from pending `can_apply=true` proposals.
- List analysis-only proposals as skipped by default.
- Publish handler name, default/max limits, idempotency-key input support, and
  explicit no-mutation safety flags.
- Extend the focused decision-queue test to pin the snapshot-only contract.

## Likely files to touch

- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/decision-queue-batch-preflight-visibility-20260606.md`

## Alternatives considered

- Run batch preflight automatically from `build_decision_queue`. Rejected
  because snapshots must remain read-only and side-effect free.
- Leave batch visibility to frontend hardcoded knowledge. Rejected because
  backend Product Layer should own the safety contract.
- Treat candidate counts as apply readiness. Rejected because preflight and
  apply remain separate supervised steps.

## Risks

Operators could mistake batch candidates for ready-to-apply proposals.
Mitigation: the summary uses batch-preflight wording, `runs_preflight=false`,
and `live_apply_entrypoint=apply_operator_proposal`.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_decision_queue_summarizes_apply_preflight_candidates -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may prefer refactoring single and batch preflight validation before adding
more summary fields. I would keep this as a small Product Layer read-model
contract and defer extraction until the handlers change again.
