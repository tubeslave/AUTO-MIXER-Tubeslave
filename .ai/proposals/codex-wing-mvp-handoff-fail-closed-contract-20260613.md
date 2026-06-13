# Codex Proposal: WING Handoff Fail-Closed Contract

## Short Thesis

Bind the operator handoff packet to the same fail-closed conditions as the
WING preflight plan, so the real hardware step exposes its stop conditions
before a backend/WING session starts.

## What I Understood

The current MVP blocker is not closed until real WING read-only artifacts are
captured and manually reviewed. Offline work should make that hardware step
more explicit and harder to misinterpret, without treating generated packets as
authority.

## Proposed Solution

Add `fail_closed_conditions` to the handoff packet, render it in markdown,
validate the packet field against the shared contract, and validate that it
matches the generated expected plan artifact.

## Likely Files to Touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- Council brief, ADR, and report

## Alternatives Considered

Keeping fail-closed conditions plan-only is smaller, but it forces the operator
to cross-open the plan to inspect the stop conditions. A larger schema refactor
is unnecessary for this narrow handoff drift guard.

## Risks

The main risk is stale handoff packets failing closed. That is acceptable
because stale packets should be rebuilt before a real WING session.

## Test Plan

Run compile checks, focused handoff/no-WING tests, no-WING quick, direction
rails, memory lint, diff check, and the standard pytest command.

## Where Kimi May Disagree

Kimi may prefer deriving the packet and plan from one schema object. For this
MVP rail, explicit field binding plus no-WING drift detection is lower risk and
easier to review.
