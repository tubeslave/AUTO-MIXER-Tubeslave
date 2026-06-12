# Proposal: no-WING frontend gate drift

## Short Thesis

Fix the frontend gate failures exposed by no-WING quick without changing runtime
safety semantics.

## What I Understood

The no-WING gate is part of the WING read-only preflight path. It should fail
only on real prestart blockers, not stale frontend type or copy expectations.

## Proposed Solution

- Add explicit `ModuleStatusRecord[]` typing where adapter rows were inferred
  too broadly.
- Keep the current canonical mode display labels in E2E expectations:
  `Analysis Mode`, `Assisted Mixing`, `Supervised AutoMix`, and
  `Full Autonomous Mixing`.

## Likely Files To Touch

- `frontend/src/services/operatorAdapters.ts`
- `frontend/e2e/automixer-prototype.spec.ts`

## Alternatives Considered

Change UI copy back to short labels. That would contradict the existing
operator adapter contract tests for display labels.

## Risks

Low. The changes are frontend type/test alignment and preserve no-unsafe-write
assertions.

## Test Plan

- `npx tsc --noEmit`
- `npm run test:e2e`
- no-WING quick probe with continue-on-failure
- Full repository pytest

## Where The Other Agent May Disagree

Kimi may prefer centralizing E2E mode labels in a shared helper. That is useful
later; this patch keeps the gate fix local.
