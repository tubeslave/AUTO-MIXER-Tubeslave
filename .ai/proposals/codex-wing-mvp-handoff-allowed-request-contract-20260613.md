# Codex Proposal: WING Handoff Allowed Request Contract

## Short Thesis

Bind the operator handoff packet to the same allowed request and request-id
contract as the WING preflight plan, so the real hardware step has one visible
read-only WebSocket boundary.

## What I Understood

The current MVP blocker is not closed until a real WING read-only artifact is
captured and manually reviewed. Offline work should make that hardware step
harder to mis-run, without treating generated packets as authority.

## Proposed Solution

Add `allowed_request_types` and `request_id_policy` to the handoff packet,
render them in markdown, validate each packet field against the shared
contract, and validate that both fields match the generated expected plan
artifact.

## Likely Files to Touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- Council brief, ADR, and report

## Alternatives Considered

Keeping these fields plan-only is smaller, but the handoff packet is the
operator's primary hardware-step artifact. Adding a broader schema generator is
larger than this focused drift guard requires.

## Risks

The main risk is stale handoff packets failing closed. That is acceptable
because stale packets should be rebuilt before a real WING session.

## Test Plan

Run compile checks, focused handoff/no-WING tests, no-WING quick, direction
rails, memory lint, diff check, and the standard pytest command.

## Where Kimi May Disagree

Kimi may prefer generating the plan and packet from one schema object. For this
MVP rail, explicit binding plus no-WING drift detection is lower risk and
easier to review.
