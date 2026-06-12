# Proposal: WING read-only handoff artifact URL

## Short Thesis

The handoff packet artifact schema should validate the URL it already records.

## What I Understood

The first MVP blocker is real WING read-only connection proof. Offline work
should only reduce drift before that hardware step and must not grant live
authority.

## Proposed Solution

Reject a handoff packet artifact when `packet["url"]` does not equal the
expected read-only backend URL.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`

## Alternatives Considered

Rely on command-generation checks alone. That protects generated packets, but
not a manually edited or stale packet artifact.

## Risks

Low. This is offline artifact validation only.

## Test Plan

- Focused handoff artifact schema tests.
- WING/no-WING adjacent tests.
- Full repository pytest.

## Where The Other Agent May Disagree

Kimi may prefer validating all packet fields against the referenced plan in one
generic diff. That is reasonable later; this patch closes the concrete URL gap.
