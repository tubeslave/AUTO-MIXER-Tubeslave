# Proposal: ProductState Promotion-Memory Queue Importer Dry-Run Blocker

## Short thesis

Run the promotion-memory queue importer only as dry-run evidence and prove the
next consumer blocks before preflight planning.

## What I understood

The ProductState promotion-memory route now emits a queue-importer request
template with `apply=false` and `operator_ack=false`. The next safe step is to
execute that template as a dry run, then verify that the queue-importer result
consumer refuses to expose a preflight-plan request until a separate applied
queue import exists.

## Proposed solution

Add narrow memory-prefixed aliases in
`backend/handlers/product_state_handlers.py` for the queue-importer-result
verifier handoff. Extend the ProductState promotion-memory regression to run
the queue importer in dry-run mode, feed that result to the existing verifier,
and assert the applied-result blocker.

## Likely files to touch

- `backend/handlers/product_state_handlers.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/reports/AUTOMIXER_PROMOTION_QUEUE_IMPORT_MEMORY_BOUNDARY_20260606.md`
- `Docs/adr/`

## Alternatives considered

Apply the queue importer in the same ProductState memory branch. Rejected
because applied queue import mutates an explicit operator queue and should
remain a separate supervised task with explicit `apply=true` and
`operator_ack=true`.

## Risks

The risk is accidentally treating dry-run importer specs as imported queue
records. The test must prove that preflight planning stays unavailable and
that queue mutation, proposal-state mutation, idempotency reservation, direct
sequence execution, live authority, and full autonomous readiness remain false.

## Test plan

Run focused ProductState and adjacent promotion consumer tests, py_compile,
memory lint, diff check, and full pytest if focused checks remain stable.

## Where the other agent may disagree

Kimi may argue that applied queue import is the more useful next proof. The
counterpoint is safety: proving the dry-run blocker first makes the authority
handoff explicit before any future queue mutation.
