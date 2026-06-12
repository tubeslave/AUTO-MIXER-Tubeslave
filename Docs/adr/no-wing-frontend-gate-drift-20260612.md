# ADR: no-WING frontend gate drift

## Context

The no-WING release gate runs frontend TypeScript and Playwright checks before
WING read-only preflight. After fixing the WING plan fail-closed mismatch, the
gate exposed stale frontend checks: row arrays in `operatorAdapters.ts` widened
`tone` to `string`, and E2E expected old short mode labels instead of the
canonical display labels.

## Options Considered

1. Change UI copy back to short mode labels.
2. Align types and E2E expectations with the current canonical labels.

## Decision

Align frontend types and E2E expectations with the current UI contract.

## Why This Won

Existing adapter contract tests define the canonical display labels. The E2E
tests should verify the current safety behavior: prototype-only mode changes,
local apply simulation, and no unsafe write requests.

## Rejected Alternatives

Changing UI copy was rejected because it would drift from the mode label
contract already used across the frontend.

## Implementation Plan

- Add explicit `ModuleStatusRecord[]` annotations in adapter row construction.
- Update E2E mode-label expectations and disambiguate the mode-changed locator.

## Test Plan

- `npx tsc --noEmit`
- `npm run test:e2e`
- no-WING quick probe with continue-on-failure
- Full repository pytest

## Risks And Rollback

The change is frontend-only and does not contact WING, backend WebSocket, or
OSC. Rollback is to restore the previous type inference and E2E strings.
