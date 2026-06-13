# Codex Proposal: WING MVP Handoff Expected Artifacts Contract

## Short Thesis

Make the handoff packet declare the same expected summary/events artifacts as
the expected plan, and make no-WING validation reject drift.

## What I Understood

The current WING MVP remains blocked on real read-only hardware proof and
manual review. Offline packet hardening may improve operator handoff quality,
but it must not contact WING, close readiness, or grant live authority.

## Proposed Solution

Add `expected_artifacts` to
`tools/build_wing_readonly_preflight_handoff_packet.py`, preserving the existing
named summary/event glob fields. Extend `tools/no_wing_release_check.py` to
compare the packet list and named fields with the shared expected globs and the
expected plan artifact list. Surface the comparison in `summary.json` and
`summary.md`.

## Likely Files to Touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- Council/report artifacts

## Alternatives Considered

Leaving only named fields is smaller, but it keeps the packet harder to compare
against the plan. Replacing the packet shape with a schema object is broader
than needed for this rail.

## Risks

Risk is limited to packet fixture and summary expectation drift. The change
does not introduce live transport, backend startup, mixer writes, or readiness
closure.

## Test Plan

Run compile, focused handoff/no-WING tests, no-WING quick, direction rails,
memory lint, diff check, and the standard pytest suite.

## Where Kimi May Disagree

Kimi may prefer removing the named summary/event fields and relying only on the
list. Keeping both is more compatible with existing reports while adding the
missing plan comparison.
