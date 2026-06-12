# Proposal: WING read-only handoff value types summary

## Short Thesis

The operator handoff path should show and validate the same value-type set as
the read-only runner plan and capture commands.

## What I Understood

The first MVP blocker is still WING read-only connection proof. Offline tooling
can only help if it reduces operator-preflight drift without granting live
authority.

## Proposed Solution

Add explicit `--value-types fader,gain,pan` to the handoff command and expose
the value-type match in no-WING evidence:

- `handoff_packet_value_types`
- `handoff_packet_value_types_match_capture`

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`

## Alternatives Considered

Keep relying on the handoff tool default. That is safe but less visible during
operator handoff and less strict in the no-WING summary.

## Risks

Low. The change is offline/report-only and adds a fail-closed mismatch check.

## Test Plan

- Focused no-WING release-check tests.
- WING MVP rails tests.
- Full repository pytest.

## Where The Other Agent May Disagree

Kimi may prefer moving all WING read-only CLI defaults into one shared contract
module. That is a reasonable later cleanup, but this patch keeps scope small.
