# ADR: WING MVP Proof Plan Operator Confirmation Bindings

## Context

The WING read-only connection proof remains the first open MVP blocker. The
read-only preflight runner fails closed when physical readback is requested
without matching `--operator-confirm-channels` and without
`--operator-confirm-session-ready`. The supervised proof plan fallback command
should expose the same operator-confirmation requirement.

## Options considered

1. Keep the proof plan fallback command unchanged.
2. Bind operator-confirmation flags directly in the proof plan fallback command.
3. Move all read-only preflight command assembly into a shared builder.

## Decision

Bind the fallback command directly to the canonical read-only preflight
channels, value types, operator channel confirmation, session-ready
confirmation, expected plan, and prestart summary. Validate and summarize those
bindings in the no-WING release gate.

## Why this won

It aligns the non-executing proof plan with the runner and runbook contract
without starting the backend, contacting WING, or broadening live authority.

## Rejected alternatives

Leaving the command unchanged preserves a stale operator path. A shared builder
is a broader refactor than this blocker needs today.

## Implementation plan

- Import canonical read-only preflight channel and value-type constants into
  `backend/supervised_wing_proof_plan.py`.
- Add operator channel/session confirmation assertions to
  `WingReadonlyPreflightPlanStep`.
- Extend `next_wing_proof_plan` summary fields and no-WING gate validation.
- Add focused tests and report validation evidence.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/supervised_wing_proof_plan.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to non-executing proof-plan and no-WING summary shape. Rollback
removes the new fields and assertions. No backend startup, WING connection,
console query, console write, readiness closure, or live authority is
introduced.
