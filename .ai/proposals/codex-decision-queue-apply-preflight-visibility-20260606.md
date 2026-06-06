# Proposal: decision-queue-apply-preflight-visibility-20260606

## Short Thesis

Add a read-only `apply_preflight` summary to the decision queue so the operator
cockpit can see which pending proposals should be dry-run checked before
supervised apply.

## What I Understood

The project now has a single-proposal apply preflight. To move toward an
autonomous mixing engineer, the queue itself should expose the next action:
which pending proposals are candidates for preflight and which are analysis-only
recommendations.

## Proposed Solution

- Add a Product Layer helper in `backend/operator_product_state.py` that
  inspects normalized pending/history proposal dictionaries.
- Count pending proposals with `can_apply=true` as preflight candidates.
- Count pending proposals without applyable requested changes as analysis-only.
- Publish candidate IDs, analysis-only IDs, handler name, live apply entrypoint,
  and no-mutation safety flags inside `decision_queue.summary`.
- Keep source-identity-specific visibility separate but compatible with the new
  generic queue visibility.

## Likely Files To Touch

- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/decision-queue-apply-preflight-visibility-20260606.md`

## Alternatives Considered

- Run preflight automatically while building the decision queue. Rejected
  because queue snapshots must remain read-only and cheap.
- Mark proposals as ready based only on `can_apply=true`. Rejected because this
  is only a preflight candidate, not a full safety result.
- Merge this into source-identity-only summary. Rejected because operator and
  manually created proposals need the same visibility.

## Risks

`can_apply=true` could be mistaken for `ready_to_apply=true`. Mitigation: name
the summary `apply_preflight`, use candidate language, and point operators to
the dry-run handler before live apply.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_decision_queue_summarizes_apply_preflight_candidates tests/test_operator_product_state.py::test_decision_queue_summarizes_source_identity_queue_import_visibility -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Where The Other Agent May Disagree

Kimi may prefer a batch preflight executor immediately. I would land queue
visibility first because it gives UI and operator workflows a stable, read-only
contract before adding a batch validator.
