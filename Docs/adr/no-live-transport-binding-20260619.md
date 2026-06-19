# ADR: No-live-transport binding

## Context

M022 introduced a standalone no-live-transport contract for offline integration
evidence. `backend/evaluation_integration_bundle_contract.py` still had a loose
`integration_checks.no_live_transport` field, so an integration bundle could
look complete without consuming the M022 contract result directly.

## Options Considered

1. Keep the loose integration check as the only no-live-transport signal.
2. Replace the loose check completely and remove it from integration checks.
3. Require a bound M022 `no_live_transport_manifest` while keeping the loose
   check as visible context.

## Decision

Require `no_live_transport_manifest` in integration bundle evaluation and run
`evaluate_no_live_transport_contract()` as a child contract before reporting
offline integration promotion readiness.

## Why This Won

This matches the existing M019 and M021 binding pattern: loose status fields can
remain visible, but they cannot satisfy a promotion gate without the bound
contract manifest. The evaluator still always returns `live_authority_granted: false`.

## Rejected Alternatives

Keeping only the loose field was too weak because a boolean does not prove
offline controls, evidence artifacts, zero runtime observations or absence of
forbidden transport tokens. Removing the loose field entirely was unnecessary
because it remains useful as summary context and regression coverage.

## Implementation Plan

- Add a no-live-transport child evaluation to
  `backend/evaluation_integration_bundle_contract.py`.
- Add blocking behavior for legacy loose `no_live_transport` checks without a
  bound M022 manifest.
- Extend integration bundle and WING MVP rail tests.
- Update roadmap, completion gate, current state and autopilot evidence.

## Test Plan

- Run `tests/test_evaluation_integration_bundle_contract.py`.
- Run `tests/test_no_live_transport_contract.py`.
- Run the M023 WING rail and full WING rail suite.
- Run the standard full pytest command before commit.

## Risks and Rollback

The main risk is treating older integration fixtures as complete even though
they lack M022 evidence. The new tests intentionally make those fixtures fail
closed. Rollback is reverting this ADR's commit; no live console state is
created or changed.
