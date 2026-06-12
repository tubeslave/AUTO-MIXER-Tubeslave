# ADR: WING read-only plan stale prestart required condition

## Context

The WING read-only runner includes
`stale_backend_port_prestart_detail_contract` in offline plan fail-closed
conditions. Backend evidence validation also expects that marker. The no-WING
release-check required list omitted it, so a generated quick-gate plan failed
artifact validation.

## Options Considered

1. Remove the stale prestart marker from runner output.
2. Add the marker to the no-WING required list.

## Decision

Add the marker to the no-WING required fail-closed condition list.

## Why This Won

The marker is part of the safer read-only prestart contract. The release gate
should accept the generated plan only when it includes that safety condition.

## Rejected Alternatives

Removing the marker was rejected because it would weaken expected-plan review
for backend-port prestart detail drift.

## Implementation Plan

- Update `tools/no_wing_release_check.py`.
- Add a test assertion for the required marker.

## Test Plan

- Focused plan artifact validation tests.
- no-WING quick gate.
- WING/no-WING adjacent tests.
- Full repository pytest.

## Risks And Rollback

The change is offline/read-only. Rollback is to remove the marker from the
required list, but that would reintroduce the quick-gate mismatch.
