# ADR: WING MVP Proof Plan Preflight Command Bindings

## Context

The WING read-only connection proof remains the first open MVP blocker. The
runbook and handoff packet require the real read-only preflight run to bind to
both a passed no-WING prestart `summary.json` and the expected offline plan.
The supervised proof plan's fallback read-only preflight step should expose the
same binding before any supervised fader/gain proof is planned.

## Options considered

1. Keep the proof plan's fallback command unchanged.
2. Add expected-plan and prestart-summary bindings to the fallback command.
3. Move all read-only preflight command assembly into a new shared builder.

## Decision

Add the expected-plan and prestart-summary bindings directly to the supervised
proof plan's read-only preflight step and make the no-WING release gate validate
and summarize them.

## Why this won

It aligns the proof plan with the runbook and handoff contract without adding a
new abstraction or touching live mixer behavior. Operators now see the same
fail-closed command shape in the plan that the real preflight path requires.

## Rejected alternatives

Leaving the command unchanged preserves a stale operator path. A shared builder
may be useful later, but it would be a broader refactor than this blocker needs.

## Implementation plan

- Add default expected-plan and prestart-summary bindings to
  `backend/supervised_wing_proof_plan.py`.
- Include those values in the `wing_readonly_preflight_required` step and
  command.
- Extend `tools/no_wing_release_check.py` to validate and summarize the fields.
- Add focused tests for the proof plan and release gate.
- Record validation in a report and link it from `Docs/INDEX.md`.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/supervised_wing_proof_plan.py tools/no_wing_release_check.py`
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
