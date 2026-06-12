# Codex Proposal: WING Read-Only Preflight Listener Confirmation Checklist

## Short thesis

Add an explicit checklist to the `backend_port_free` fail-closed artifact so an
operator can distinguish an intended backend from an unexpected listener before
starting real WING read-only preflight.

## What I understood

The current first MVP blocker is still `WING read-only connection proof`.
`tools/no_wing_release_check.py --quick` can fail when port `8765` already has
a listener. The artifact lists process details and says operator confirmation
is required, but it does not expose a structured checklist of the confirmation
steps.

## Proposed solution

- Add `listener_confirmation_checklist` to the shared prestart failure-detail
  contract.
- Emit that checklist from failed `backend_port_free` checks.
- Render it in Markdown summaries.
- Update the WING read-only preflight runbook and regression tests.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tests/test_no_wing_release_check.py`

## Alternatives considered

- Leave the checklist as prose in `operator_next_step`. Rejected because the
  handoff and summary artifacts are easier to validate when the checklist is a
  stable field.
- Automatically stop the listener. Rejected because the gate must remain
  diagnostic only and must not mutate local/backend state.

## Risks

The checklist can become stale if backend startup conventions change. Keeping
it in the shared prestart contract makes drift visible in tests.

## Test plan

- Focused backend-port/listener/no-WING tests.
- Real local no-WING quick run to inspect current busy-port summary.
- Adjacent WING/no-WING tests, memory lint, and full pytest before commit.

## Where the other agent may disagree

Kimi may prefer a shorter artifact. I would keep the checklist because the
current hardware step is operator-driven and false confidence around a busy
backend port is riskier than slightly more verbose evidence.
