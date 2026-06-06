# Proposal: source-identity-queue-import-handler-20260606

## Short Thesis

Expose source identity queue import through a Product Layer handler that stays
dry-run by default and only mutates the native operator queue after explicit
operator acknowledgement.

## What I Understood

The importer boundary exists, but operators need a callable backend action to
inspect and later apply it. This action must not become a live console path.

## Proposed Solution

- Add `import_source_identity_queue_ordering_plan` to
  `product_state_handlers.register_handlers`.
- If explicit plan/payloads are not supplied, build the current main pipeline
  preview and use its queue-ordering plan plus proposal payloads.
- Call `build_source_identity_proposal_queue_importer`.
- Pass `server.operator_proposal_queue` only when `apply=true` and
  `operator_ack=true`.
- Return the importer payload to the client.

## Likely Files To Touch

- `backend/handlers/product_state_handlers.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-queue-import-handler-20260606.md`

## Alternatives Considered

- Import automatically during main pipeline preview. Rejected because preview
  builders must remain non-mutating.
- Add a new `server.py` route. Rejected because `server.py` is a coordinator and
  Product Layer handlers already own these actions.
- Call supervised apply after import. Rejected because queue import and console
  apply are separate operator decisions.

## Risks

- Operators may conflate queue import with console apply. Mitigate by keeping
  imported entries in the queue and leaving live writes behind existing apply
  approval/idempotency gates.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_source_identity_proposal_queue_importer.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/source_identity_proposal_queue_importer.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi may prefer waiting for frontend controls first. I would land the backend
handler now because it gives a testable operator boundary for future UI work
without adding live writes.
