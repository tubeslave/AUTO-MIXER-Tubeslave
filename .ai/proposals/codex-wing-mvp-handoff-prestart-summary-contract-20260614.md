# Codex Proposal: WING MVP Handoff Prestart Summary Contract

## Short thesis

Make the WING read-only handoff packet carry and validate the same prestart
summary contract that the offline plan artifact already requires.

## What I understood

The current first blocker is still `WING read-only connection proof`. We can move
the MVP forward only by tightening offline evidence and handoff gates until a
real WING read-only run is available. The handoff packet currently carries the
prestart summary path and SHA binding, but it should also declare the expected
summary type, status, connection policy, no-contact flags, and backend check
status.

## Proposed solution

- Add the contract fields to `tools/build_wing_readonly_preflight_handoff_packet.py`.
- Render those fields in the operator packet Markdown.
- Extend `tools/no_wing_release_check.py` handoff packet validation so stale or
  unsafe prestart summary contract fields fail closed.
- Add positive and negative tests in the existing handoff/no-WING suites.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- Council trail docs for this slice

## Alternatives considered

1. Leave the fields only in the plan artifact. This is weaker because operators
   and packet validators can miss a stale packet even when SHA/path are present.
2. Move all validation into the runner. That is too late for this gate because
   the handoff packet is designed to block before a backend/WING session starts.

## Risks

- Adding many fields could create maintenance noise. The risk is acceptable
  because all values come from shared contract constants and match existing
  plan-artifact fields.
- This still does not prove real WING connectivity. The blocker must remain open.

## Test plan

- Run focused handoff/no-WING pytest.
- Generate a handoff packet and inspect the new fields.
- Run no-WING quick and confirm fail-closed behavior when the backend listener is
  already present.
- Run the full standard pytest command before commit.

## Where the other agent may disagree

Kimi may argue the handoff packet is already covered by plan validation. My
counterpoint is that the packet is the operator-facing gate, so it should be
self-describing and stale packet drift should fail at the packet validator.
