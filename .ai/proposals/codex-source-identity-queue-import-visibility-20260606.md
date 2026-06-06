# Proposal: source-identity-queue-import-visibility-20260606

## Short Thesis

Add read-only source identity queue-import visibility to the Product Layer
decision queue, so imported proposals are explainable before any operator
chooses to apply them.

## What I Understood

The queue importer now has a guarded Product Layer handler. The next autonomy
step is not broader live execution; it is making the imported queue state
auditable: provenance, rank source, applyability, and safety envelope should be
visible in one compact decision-queue summary.

## Proposed Solution

- Add a helper in `backend/operator_product_state.py` that inspects normalized
  pending/history proposal dictionaries.
- Identify imported proposals by `source_identity_queue_import` metadata or by
  `source=source_identity_proposal_queue_importer`.
- Count pending imported proposals, ranked/unranked imported proposals,
  applyable pending proposals, and analysis-only pending proposals.
- Attach a nested `source_identity_queue_import` summary plus stable top-level
  count aliases inside `decision_queue.summary`.
- Keep safety fields read-only and explicit: source identity import itself has
  `live_write=false` and `can_apply_live=false`; console apply remains behind
  `apply_operator_proposal`.

## Likely Files To Touch

- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-queue-import-visibility-20260606.md`

## Alternatives Considered

- Add the summary to `OperatorProposalQueue.summary()`. Rejected because this is
  Product Layer visibility, not queue storage behavior.
- Add a new handler just for source identity queue status. Rejected because the
  existing decision queue already owns operator queue visibility.
- Promote imported proposals directly into live apply. Rejected because live
  write authority remains restricted to the existing supervised fader/gain path.

## Risks

Operators may treat `can_apply=true` on a gain proposal as source identity live
authority. Mitigation: the source identity import summary explicitly says import
is not live apply and later console writes remain governed by the existing
proposal apply gate.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_decision_queue_summarizes_source_identity_queue_import_visibility tests/test_operator_product_state.py::test_source_identity_queue_import_handler_imports_with_ack_without_live_write -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi may prefer a dedicated queue-import status endpoint. I would keep this in
`build_decision_queue` because the imported proposals already live there and the
UI should not need a second call to understand queue provenance.
