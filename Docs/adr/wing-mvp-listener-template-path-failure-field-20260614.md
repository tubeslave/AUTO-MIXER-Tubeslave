# ADR: WING MVP Listener Template Path Failure Field

## Context

The no-WING backend-port prestart gate generates a backend-listener
confirmation template when it finds an existing listener on the intended
backend port. The failed summary exposes `listener_confirmation_template_path`
so the operator can inspect and edit the template, then rerun the gate with
`--backend-listener-confirmation`.

The shared `backend_port_prestart_check_failure_detail_fields` contract did not
require this path. That left a stale-artifact gap: a summary could satisfy the
field-list contract while omitting the artifact path needed for operator
review.

## Options considered

1. Keep `listener_confirmation_template_path` as a prose-only rail.
2. Add a one-off validator for the path.
3. Add `listener_confirmation_template_path` to the shared failure-detail field
   contract.

## Decision

Add `listener_confirmation_template_path` to
`WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS`.

## Why this won

The path is part of the fail-closed operator workflow, not optional decoration.
Putting it in the shared list makes stale no-WING summaries, plans, handoff
packets, and reports fail closed if they omit the artifact path. The change
does not touch live WING behavior or grant readiness.

## Rejected alternatives

Keeping the path prose-only keeps the same mechanical-contract gap. A one-off
validator would duplicate the existing list-based staleness checks.

## Implementation plan

- Extend the shared backend-port prestart failure-detail field list.
- Update WING MVP rails and autoloop docs.
- Add explicit tests for constants, Project Director output, generated
  progress, and docs index coverage.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_supervised_wing_proof_plan.py tests/test_autonomy_promotion_readiness.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to stale artifact contract failures. Rollback removes the path
from the shared list and rail docs, then regenerates latest progress. No live
WING, backend, or readiness state is changed.
