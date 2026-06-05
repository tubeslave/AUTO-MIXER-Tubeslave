# ADR: proof-evidence-settings-visibility-20260605

## Context

The Product Layer exposes a read-only supervised proof evidence snapshot and
sanitized autonomy evidence overlay rows. The frontend adapter maps these into
module status rows, but browser-level Settings coverage still needs proof that
operators can see proof source freshness and target-level proof blockers.

## Options considered

1. Extend the existing Settings module status surface.
2. Build a new proof evidence page.
3. Keep proof visibility adapter-only for now.

## Decision

Use option 1. Settings will summarize proof evidence freshness in the inspector
and expose a Proof filter in the module status table. E2E backend DTO coverage
will include the proof evidence payload and visible assertions.

## Why this won

It improves operator explainability with a small UI change and no new live write
path. The existing Settings table already has the right shape for source status,
target rows, search, and review-only next steps.

## Rejected alternatives

A new page would be premature and larger than this evidence slice. Adapter-only
coverage is too weak because it does not prove the actual browser-visible
operator path.

## Implementation plan

1. Derive proof evidence summary rows from `moduleStatuses`.
2. Keep all proof rows read-only and separate source freshness from target
   overlay blockers.
3. Add a Proof filter to the module status table.
4. Extend the backend DTO e2e fixture with supervised proof evidence and
   autonomy evidence overlay rows.
5. Assert Settings renders source status, freshness text, and target overlay
   rows while the no-live-write command assertions remain in place.

## Test plan

- `node frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- Playwright e2e Settings/backend DTO path when local dependencies are
  available.

## Risks and rollback

Risk: fresh proof evidence could be mistaken for autonomy approval. The UI keeps
the source row read-only and displays target-level blockers separately.

Rollback is removing the Settings summary/filter, e2e fixture additions, and this
ADR.
