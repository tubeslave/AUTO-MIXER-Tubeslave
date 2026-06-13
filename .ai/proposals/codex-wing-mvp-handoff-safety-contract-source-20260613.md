# Codex Proposal: WING MVP Handoff Safety Contract Source

## Short Thesis

Move WING read-only handoff safety lists into the shared preflight contract so
the handoff packet and no-WING validator cannot drift before real hardware
proof.

## What I Understood

The current MVP is still the supervised WING Rack proof, with `WING read-only
connection proof` first. Offline handoff packets must remain non-authoritative:
they must forbid live request types and explicitly say which blockers remain
open. These lists are safety contract data, not release-check implementation
details.

## Proposed Solution

Add `WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES` and
`WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS` to
`backend/wing_readonly_preflight_contract.py`. Update
`tools/no_wing_release_check.py` to keep compatibility aliases from the shared
source, and update `tools/build_wing_readonly_preflight_handoff_packet.py` to
consume the shared constants directly. Register the shared contract file in the
existing lab-only proof/release write-surface inventory entry because it
intentionally contains forbidden live-write request names as audit strings.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/write_surface_inventory.py`
- `tools/no_wing_release_check.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- focused static-scan/handoff/no-WING tests
- WING MVP report/ADR/index/evidence docs

## Alternatives Considered

Leaving the lists in `tools/no_wing_release_check.py` keeps today's behavior
but preserves tool-to-tool coupling. Moving only the builder import would not
solve ownership. The shared preflight contract is the correct boundary because
these values define packet safety, not command execution.

## Risks

The main risk is changing list order or omitting a forbidden request. Focused
tests should compare exact generated packet values and release-check aliases to
the shared contract.

## Test Plan

- Compile touched modules.
- Run focused static-scan/handoff/no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run the full pytest suite.

## Where The Other Agent May Disagree

Kimi may argue that this is another narrow refactor. The counterpoint is that
these fields are direct live-authority denials in the first hardware operator
packet, so a single source lowers the chance of a bad handoff without widening
scope.
