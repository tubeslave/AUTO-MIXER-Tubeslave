# Codex Proposal: WING MVP Handoff Proof Identity Contract

## Short Thesis

Bind WING handoff packets to the same proof identity as the expected plan so an
operator packet cannot drift from the schema, runner, or current MVP blocker it
is supposed to represent.

## What I Understood

The current MVP remains blocked on real WING read-only proof and manual review.
Offline packet hardening may improve handoff quality, but it must not close
readiness, contact WING, or grant live authority.

## Proposed Solution

Add `proof_artifact_schema`, `proof_runner`, `selected_mvp_blocker`, and
`selected_mvp_blocker_status` to
`tools/build_wing_readonly_preflight_handoff_packet.py`. Extend
`tools/no_wing_release_check.py` to compare those packet fields against both
`backend.wing_readonly_preflight_contract` and the expected plan artifact
snapshot. Surface the comparison in `summary.json` and `summary.md`.

## Likely Files to Touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- Council/report artifacts

## Alternatives Considered

Keeping proof identity only inside the plan is smaller, but it leaves the
handoff packet less self-auditing. Replacing the packet with a schema object is
broader than needed for this rail.

## Risks

Risk is limited to no-WING validation and fixture expectations. The change does
not introduce live transport, backend startup, mixer writes, or readiness
closure.

## Test Plan

Run compile, focused handoff/no-WING tests, no-WING quick, direction rails,
memory lint, diff check, and the standard pytest suite.

## Where Kimi May Disagree

Kimi may prefer a single schema model for plan and packet. For this MVP step,
explicit packet fields plus drift checks keep the diff small and make the
operator handoff clearer.
