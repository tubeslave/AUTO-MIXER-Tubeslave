# Proposal: wing-readonly-preflight-handoff-forbidden-request-coverage-20260607

## Short thesis

Make the read-only WING preflight handoff packet safer by explicitly listing the
full forbidden live request surface, make the no-WING artifact gate enforce
that list, and keep the plan-only fail-closed contract aligned with event-log
WING-mode validation.

## What I understood

The selected Project Director task is still `WING read-only connection proof`.
Hardware is not being contacted in this cycle, so useful work must prepare the
real read-only proof without pretending to close it.

## Proposed solution

1. Expand `FORBIDDEN_LIVE_REQUEST_TYPES` in
   `tools/build_wing_readonly_preflight_handoff_packet.py`.
2. Add the same required forbidden request contract to
   `tools/no_wing_release_check.py`.
3. Add the handoff packet builder to write-surface inventory as lab-only proof
   tooling, because the expanded forbidden list intentionally contains write
   token strings.
4. Add `event_log_wing_mode_not_proven` to the offline preflight plan and
   release-check required fail-closed conditions.
5. Update focused tests so a missing `set_gain` entry fails closed and the plan
   names the event-log WING-mode proof boundary.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tools/wing_readonly_preflight_runner.py`
- `backend/write_surface_inventory.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `Docs/adr/wing-readonly-preflight-handoff-forbidden-request-coverage-20260607.md`

## Alternatives considered

- Leave the shorter list: rejected because the operator handoff would not name
  the full write surface that must stay absent from read-only proof artifacts.
- Import the runner's forbidden set: rejected to keep the packet builder simple
  and transport-free by source inspection.

## Risks

The list can drift if new live request types are added elsewhere. The no-WING
gate now makes that drift visible for the current handoff artifact.

## Test plan

- Focused handoff packet tests.
- Focused no-WING handoff artifact pass/fail tests.
- Runner plan-only test to keep event-log WING-mode fail-closed coverage aligned.
- Static write-surface inventory scan.
- Full repository pytest before commit.

## Where the other agent may disagree

An independent reviewer may prefer a single shared constants module. For this
small safety slice, duplication is acceptable because it avoids adding imports
between operator tooling and release validation.
