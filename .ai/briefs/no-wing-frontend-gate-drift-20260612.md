# Brief: no-WING frontend gate drift

## Context

After the WING read-only plan mismatch was fixed, `no_wing_release_check --quick
--continue-on-failure` exposed frontend gate drift: TypeScript failed in
`operatorAdapters.ts`, and Playwright expected old short mode labels.

## Task

Restore frontend TypeScript and E2E checks without changing live-write
behavior.

## Acceptance

- `npx tsc --noEmit` passes.
- `npm run test:e2e` passes.
- E2E keeps checking that prototype mode changes and applies remain local and
  do not emit unsafe writes.
- No WING/backend/OSC contact is introduced.
