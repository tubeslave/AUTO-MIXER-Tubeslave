# Proposal: runner environment-blocked prestart summary

## Short thesis

Keep the read-only runner fail-closed on environment-blocked no-WING summaries,
but surface the specific environment-blocked reason and operator next step.

## What I understood

The active WING MVP blocker is still `WING read-only connection proof`. Before a
real read-only preflight can open an existing backend/WING session, the runner
requires a passed no-WING prestart summary. A failed no-WING summary caused by
frontend bind denial is not proof, but it should remain diagnosable when the
runner rejects it.

## Proposed solution

- Preserve `environment_blocked_checks` and `environment_blocked_check_details`
  while loading the prestart summary.
- Add `prestart_summary_environment_blocked` ahead of the generic
  `prestart_summary_status_not_passed` violation.
- Give that violation a specific operator next step that names
  `127.0.0.1:3000` and states that no readiness or live authority is granted.
- Add a regression test proving the runner blocks before WebSocket.

## Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `.ai/briefs/wing-mvp-runner-environment-blocked-prestart-summary-20260614.md`
- `.ai/proposals/codex-wing-mvp-runner-environment-blocked-prestart-summary-20260614.md`
- `Docs/adr/wing-mvp-runner-environment-blocked-prestart-summary-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_RUNNER_ENVIRONMENT_BLOCKED_PRESTART_SUMMARY_20260614.md`

## Alternatives considered

- Leave the runner with the generic invalid-prestart message: rejected because
  it hides the actionable sandbox/frontend-bind cause.
- Treat an environment-blocked no-WING summary as usable prestart proof:
  rejected because the summary is failed and cannot support WING readiness.
- Add a wider no-WING artifact parser: rejected because this step only needs
  the existing environment-blocked fields.

## Risks

The new violation must not replace the existing `status_not_passed` block. It
is additive and keeps the generic status violation in place.

## Test plan

- Focused `tests/test_wing_readonly_preflight_runner.py`.
- Focused WING MVP rails/Project Director/no-WING group.
- `py_compile`, memory lint, diff check, and full standard suite.

## Where the other agent may disagree

Kimi may prefer not adding runner-specific handling for a frontend check. The
counterargument is that the runner is the first physical-session boundary, so
its blocked summary should preserve the reason that prevents the prestart proof
from being accepted.
