# Proposal: ProductState Promotion-Memory Applied Queue Import Preflight Request

## Short thesis

Allow the ProductState promotion-memory route to prove supervised queue import
as source evidence, then keep the downstream verifier read-only while emitting
the preflight-plan request.

## What I understood

The promotion-memory route already proves that dry-run importer specs cannot
unlock preflight planning. The next useful step is to apply the queue importer
under explicit operator acknowledgement, verify that source evidence, and
expose only a read-only preflight-plan request template.

## Proposed solution

Extend the ProductState promotion-memory regression to re-run the existing
queue-importer request with `apply=true` and `operator_ack=true`, using a
fader-shaped proposal payload. Assert the queue import mutates only the
operator proposal queue and never calls mixer transport. Then feed the applied
importer result into the existing queue-importer result consumer and assert the
preflight-plan request template plus read-only consumer flags.

## Likely files to touch

- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/reports/AUTOMIXER_PROMOTION_QUEUE_IMPORT_MEMORY_BOUNDARY_20260606.md`
- `Docs/adr/`

## Alternatives considered

Stop at the dry-run blocker. Rejected because the applied importer is the
required source evidence before preflight planning. Skip to running the
preflight plan immediately. Rejected for this slice because the request
boundary should be reviewed first.

## Risks

The risk is confusing applied queue-import source mutation with consumer-side
authority. The test and docs must state that the importer mutates only the
operator proposal queue under explicit acknowledgement, while the verifier
keeps idempotency, proposal status/order mutation, direct sequence execution,
and live authority false.

## Test plan

Run focused ProductState and adjacent promotion consumer tests, py_compile,
memory lint, diff check, and full pytest if focused checks remain stable.

## Where the other agent may disagree

Kimi may argue that queue mutation should remain out of the ProductState
promotion-memory path. The counterpoint is that the queue importer is already
operator-gated and non-live; proving this source evidence is necessary before
the autonomous chain can reach preflight planning safely.
