# Proposal: WING read-only handoff URL and artifact root

## Short Thesis

The operator handoff command should make URL and artifact-root parity explicit
before the real WING read-only preflight.

## What I Understood

The current blocker is still real WING read-only connection proof. Offline work
is useful only when it reduces drift in the hardware handoff path and keeps
live authority locked.

## Proposed Solution

Pass `--url ws://localhost:8765` and
`--artifact-root artifacts/wing_readonly_preflight` through the handoff command,
then expose match booleans in no-WING evidence.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`

## Alternatives Considered

Keep relying on handoff defaults. That is safe but leaves key expected-plan
fields less visible to the operator.

## Risks

Low. This is offline/report-only validation and does not contact WING or grant
live-write authority.

## Test Plan

- Focused no-WING release-check tests.
- WING MVP rails tests.
- Full repository pytest.

## Where The Other Agent May Disagree

Kimi may prefer a shared command builder used by runner, handoff, and no-WING
checks. That is a reasonable refactor later, but this patch keeps scope tight.
