# ADR: WING MVP Selected Blocker Contract

## Context

The WING read-only preflight plan carries `selected_mvp_blocker` and
`selected_mvp_blocker_status` fields. They identify the current hardware proof
target and must match evidence intake and no-WING validation.

## Options Considered

1. Keep duplicated strings.
2. Move the selected blocker name and status into shared contract constants.
3. Parse the blocker table dynamically inside the runner.

## Decision

Use shared constants in `backend/wing_readonly_preflight_contract.py`.

## Why This Won

It removes producer/consumer drift without changing artifact shape or adding a
runtime dependency on markdown parsing.

## Rejected Alternatives

Duplicated strings were rejected because they can drift. Dynamic parsing was
rejected because the capture runner should stay deterministic and small.

## Implementation Plan

Add contract constants, consume them in runner/evidence/no-WING validation, and
update tests/fixtures.

## Test Plan

Run focused WING/no-WING tests, no-WING quick, rails, guards, memory lint, diff
check, and the full pytest suite.

## Risks and Rollback

Risk is limited to stale contract values if the blocker changes. Roll back this
patch if validation shows the selected blocker source is wrong; no live-write
capability is introduced.
