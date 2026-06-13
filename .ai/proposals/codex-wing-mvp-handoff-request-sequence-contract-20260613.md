# Codex Proposal: WING Handoff Request Sequence Contract

## Short Thesis

Bind the operator handoff packet to the exact read-only request sequence used
by the WING preflight plan, so the real hardware step can be inspected without
cross-opening generated plan JSON.

## What I Understood

The current MVP blocker is not closed until real WING read-only artifacts are
captured and manually reviewed. Offline work should make the hardware step
more explicit and harder to mis-run, without treating packets as authority.

## Proposed Solution

Add `request_sequence` and `request_count` to the handoff packet, render the
sequence in markdown, validate the packet sequence against the generated plan
artifact, and validate it against the expected sequence computed by no-WING
plan validation.

## Likely Files to Touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- Council brief, ADR, and report

## Alternatives Considered

Keeping request sequence plan-only is smaller, but the handoff packet is the
operator's primary hardware-step artifact. A larger schema generator is
unnecessary for this focused drift guard.

## Risks

The main risk is stale handoff packets failing closed. That is acceptable
because stale packets should be rebuilt before a real WING session.

## Test Plan

Run compile checks, focused handoff/no-WING tests, no-WING quick, direction
rails, memory lint, diff check, and the standard pytest command.

## Where Kimi May Disagree

Kimi may prefer moving request-address generation into the shared contract. For
this MVP rail, a small packet-local sequence builder plus no-WING drift
detection is lower risk and easier to review.
