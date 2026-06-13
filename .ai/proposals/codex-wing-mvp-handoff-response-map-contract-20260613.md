# Codex Proposal: WING Handoff Response Map Contract

## Short Thesis

Bind the operator handoff packet to the same read-only response-type contract as
the WING preflight plan, so the hardware step has one visible request/response
shape across preview, capture, and review.

## What I Understood

The current MVP blocker is not closed until a real WING read-only artifact is
captured and manually reviewed. Offline work should make that hardware step
safer and harder to misread, without turning candidate artifacts into authority.

## Proposed Solution

Add `expected_response_types_by_request_type` to the handoff packet, render it
in markdown, validate the packet field against the shared contract, and validate
that it matches the generated expected plan artifact.

## Likely Files to Touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- Council brief, ADR, and report

## Alternatives Considered

Keeping the map plan-only is smaller, but it forces the operator to cross-open
the plan to verify response shapes. A larger schema refactor is unnecessary for
this narrow handoff drift guard.

## Risks

The main risk is stale test fixtures or packet artifacts failing closed. That
is acceptable because a stale handoff packet should be rebuilt before hardware
use.

## Test Plan

Run compile checks, focused handoff/no-WING tests, no-WING quick, direction
rails, memory lint, diff check, and the standard pytest command.

## Where Kimi May Disagree

Kimi may prefer putting all packet/plan fields behind one schema generator. For
this MVP rail, explicit field binding plus no-WING drift detection is lower
risk and easier to review.
