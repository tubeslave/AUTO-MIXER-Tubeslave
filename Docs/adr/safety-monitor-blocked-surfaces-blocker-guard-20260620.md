# ADR: Safety Monitor Blocked Surfaces Blocker Guard

## Context

M056 and M057 made the Safety Monitor `Policy` and `Goal readiness` rows fail
closed while offline integration bundle blockers are visible. The same panel
still rendered the footer as `Blocked: none` when `safety.blockedSurfaces` was
empty, even while global promotion planning was blocked.

## Options Considered

1. Leave the footer unchanged because other panels already show the blocker.
2. Derive local blocked-surface presentation from
   `integrationBundleApplyBlocked`, while preserving the raw safety blocked
   surfaces when the blocker is absent.
3. Mutate `safety.blockedSurfaces` before rendering the panel.

## Decision

Add local Safety Monitor blocked-surface presentation state derived from
`integrationBundleApplyBlocked`. When the global blocker is visible, the footer
includes `offline_integration_bundle_blockers_visible` before any existing
safety blocked surfaces. When the blocker is absent, the existing footer
behavior remains unchanged.

This does not mutate safety monitor input, readiness input, backend status
input, backend result input, task data, queue proposals, open Supervised Apply,
dispatch supervised apply, contact backend/WING, grant live authority or close
readiness.

## Why This Won

The Safety Monitor footer is the compact blocked-surface summary. Showing
`Blocked: none` while a global offline integration blocker is active creates a
contradictory safety surface. A local presentation guard keeps the footer
aligned with the global blocker without altering backend or adapter evidence.

## Rejected Alternatives

Leaving the footer unchanged preserved ambiguity on a safety surface. Mutating
`safety.blockedSurfaces` was rejected because the guard is presentation-only and
must not rewrite backend-derived safety monitor input.

## Implementation Plan

1. Add local `blockedSurfaces` in `SafetyMonitorPanel`.
2. Prefix `offline_integration_bundle_blockers_visible` only while
   `integrationBundleApplyBlocked` is true.
3. Preserve the existing blocked-surface list and `none` fallback when the
   blocker is absent.
4. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m058_safety_monitor_blocked_surfaces_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Safety Monitor footer presentation. If it regresses,
revert the local blocked-surface presentation guard and M058 evidence/test/docs
updates. The rollback does not alter backend gates, WING transport or accepted
hardware evidence.
