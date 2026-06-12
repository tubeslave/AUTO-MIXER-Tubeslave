# Proposal: WING read-only handoff value types

## Short Thesis

The offline handoff packet should fail closed on the same value-type contract as
the read-only runner.

## What I Understood

The current MVP waits for real WING read-only proof. Tooling can still improve
the operator path if it directly reduces preflight drift and remains offline.

## Proposed Solution

Validate handoff `value_types` during packet construction:

- require at least one value type;
- reject duplicates;
- allow only `fader`, `gain`, and `pan`.

## Likely Files To Touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`

## Alternatives Considered

Rely on the runner to reject invalid value types later. That keeps behavior safe
but makes the operator packet less trustworthy.

## Risks

Low. The change is offline and only rejects invalid handoff input earlier.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py`
- Adjacent WING/no-WING tests.

## Where The Other Agent May Disagree

Kimi may prefer moving the allowed value-type set into a shared contract module.
That is reasonable later, but this patch keeps the diff small and matches the
runner's existing public contract.
