# Proposal: WING read-only plan stale prestart required condition

## Short Thesis

The no-WING release gate must require the same fail-closed plan markers that the
runner generates.

## What I Understood

`tools/no_wing_release_check.py --quick` generated a WING read-only plan, then
rejected that plan because its required fail-closed list was stale. The runner
and backend evidence contract already require the stale prestart detail marker.

## Proposed Solution

Add `stale_backend_port_prestart_detail_contract` to
`WING_READONLY_PREFLIGHT_PLAN_REQUIRED_FAIL_CLOSED_CONDITIONS`.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`

## Alternatives Considered

Remove the marker from runner output. That would weaken a documented prestart
gate and contradict backend evidence validation.

## Risks

Low. This aligns validation with existing generated artifacts and does not
contact hardware.

## Test Plan

- Focused plan artifact validation tests.
- no-WING quick gate.
- WING/no-WING adjacent tests.
- Full repository pytest.

## Where The Other Agent May Disagree

Kimi may prefer importing the backend evidence constant instead of duplicating
the tuple. That is a useful cleanup later; this patch keeps release-check import
surface stable.
