# ADR: Require Expected Plan For Real WING Read-Only Preflight

## Context

The supervised WING Rack MVP must first prove a real read-only connection and
physical console readback. The offline preflight plan is the contract that binds
operator-approved channel scope and request sequence before hardware contact.

## Options Considered

1. Leave `--expected-plan` optional and reject missing proof during artifact
   intake.
2. Require `--expected-plan` in `argparse` for all modes.
3. Require `--expected-plan` only for non-`--plan-only` runs, before WebSocket.

## Decision

Use option 3. A real run without `--expected-plan` fails closed locally before
opening WebSocket.

## Why This Won

It enforces the hardware-contact boundary exactly where the runbook says it
must hold, while preserving offline plan generation. It also leaves an audit
summary explaining why the run was blocked.

## Rejected Alternatives

- Artifact-intake-only rejection is too late for the first hardware step.
- Global parser enforcement would break the mode that creates the plan.

## Implementation Plan

- Add a missing-expected-plan branch in the real runner path before
  `websockets.connect`.
- Add a regression test proving WebSocket is not opened.
- Update existing fake WebSocket tests to provide the expected plan.
- Clarify the runbook.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_require_expected_plan_20260607`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

The main risk is operator friction for older commands. Rollback would revert the
guard, but that would weaken the first WING proof boundary and should require a
new safety decision.
