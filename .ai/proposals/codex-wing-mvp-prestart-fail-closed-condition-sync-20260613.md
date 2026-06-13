# Codex Proposal: WING MVP Prestart Fail-Closed Condition Sync

## Short Thesis

Align the no-WING release-check expected fail-closed list with the WING readonly
preflight runner and evidence contract so the prestart summary fail-closed
conditions are validated consistently.

## What I Understood

The WING readonly offline plan must fail closed if its no-WING prestart summary
is missing, invalid, or SHA-mismatched. The runner and evidence validator already
encode this, but the release-check validator was comparing against a stale list.

## Proposed Solution

Add `prestart_summary_missing_or_invalid` and
`prestart_summary_sha256_mismatch` to
`WING_READONLY_PREFLIGHT_PLAN_REQUIRED_FAIL_CLOSED_CONDITIONS` in
`tools/no_wing_release_check.py`, and assert those missing-condition violation
names in the unsafe-plan test.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- WING MVP docs, report index, blocker evidence, and ADR/report artifacts

## Alternatives Considered

Relaxing the equality check would avoid the mismatch, but would also allow
stale or unexpected fail-closed contracts. Keeping strict equality is safer for
preflight gates.

## Risks

Older plan fixtures without the two prestart conditions will now fail release
validation. That is desired because accepted WING readonly plans must include
the current prestart summary safety contract.

## Test Plan

- Focused `test_no_wing_release_check.py`.
- Focused WING readonly runner/evidence tests.
- no-WING quick on a free backend-port.
- Project Director and direction guard.
- Full pytest.

## Where The Other Agent May Disagree

Kimi may prefer importing the runner/evidence constant directly to prevent
future duplication. For this narrow fix, updating the local release-check
constant preserves the current standalone release-check shape without changing
module coupling.
