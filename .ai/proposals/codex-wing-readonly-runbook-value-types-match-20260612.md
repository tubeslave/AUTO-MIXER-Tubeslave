# Proposal: Align WING runbook value types

## Short Thesis

The WING read-only runbook should show the same value types as the no-WING
prestart gate: `fader,gain,pan`.

## What I Understood

The expected-plan runner is intentionally strict. If the operator copies a
runbook command that differs from the no-WING plan/capture template, the real
preflight can fail before WebSocket for the wrong reason.

## Proposed Solution

- Update both plan-only and real-run examples in
  `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.
- Add a rails assertion that the runbook contains `fader,gain,pan` and not the
  stale `fader,gain` example.

## Likely Files To Touch

- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tests/test_wing_mvp_rails.py`

## Alternatives Considered

Leaving the examples at `fader,gain` was rejected because it creates command
drift from the checked no-WING contract.

## Risks

This adds a read-only pan readback target to the copied example. It does not
send writes, proposals, or live authority.

## Test Plan

- Focused rails test.
- Adjacent WING/no-WING tests.
- Full pytest before commit.

## Where The Other Agent May Disagree

Kimi may prefer narrowing to fader/gain only. The project's current runner
default and release gate already include pan, so the safer action is to align
the operator-facing command with the enforced contract.
