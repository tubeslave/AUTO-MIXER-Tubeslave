# ADR: WING MVP Prestart Fail-Closed Condition Sync

## Context

The supervised WING Rack MVP waits on real `WING read-only connection proof`.
Before any real backend/WING session, the offline expected plan must be validated
without contacting hardware. The runner and evidence validator both require the
plan to fail closed when its no-WING prestart summary is missing or SHA
mismatched, but the release-check validator still used an older fail-closed
condition list.

## Options Considered

1. Relax release-check fail-closed list equality.
2. Import the runner/evidence condition list into the release-check module.
3. Update the release-check required list to include the two prestart summary
   conditions.

## Decision

Use option 3. The release-check required list now includes
`prestart_summary_missing_or_invalid` and
`prestart_summary_sha256_mismatch`.

## Why This Won

Strict equality remains useful because it catches stale or widened preflight
plans. Adding the missing conditions aligns the validator with the current
runner/evidence contract without changing module boundaries.

## Rejected Alternatives

Relaxed matching was rejected because unexpected conditions should still fail
review. Importing the runner list was deferred because `no_wing_release_check`
currently keeps a standalone release gate surface.

## Implementation Plan

- Add the two prestart conditions to
  `WING_READONLY_PREFLIGHT_PLAN_REQUIRED_FAIL_CLOSED_CONDITIONS`.
- Extend the unsafe-plan test to assert missing-condition violations for both
  prestart summary entries.
- Validate with focused and full tests plus no-WING quick.

## Test Plan

- Focused no-WING and WING readonly test slices.
- no-WING quick on an unused backend port.
- Project Director and direction guard.
- Full pytest.

## Risks And Rollback

The change may reject old expected-plan artifacts. Rollback would remove the two
conditions from release-check validation, but that would reintroduce the stale
prestart summary contract drift.
