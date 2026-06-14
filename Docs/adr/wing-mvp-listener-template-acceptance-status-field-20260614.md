# ADR: WING MVP Listener Template Acceptance Status Field

## Context

The no-WING backend-port prestart gate fails closed when a listener is already
present on the backend port. In that path it generates a listener-confirmation
template and emits `listener_confirmation_template_acceptance_status`.

The required `backend_port_prestart_check_failure_detail_fields` contract did
not list that field, even though the operator must know whether the generated
template is still pending review before any WING read-only session can proceed.

## Options considered

1. Keep the acceptance status as an optional output field.
2. Add separate ad hoc checks for the acceptance status.
3. Add `listener_confirmation_template_acceptance_status` to the shared
   failure-detail field contract.

## Decision

Add `listener_confirmation_template_acceptance_status` to
`WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS`.

## Why this won

It keeps the busy-port fail-closed path self-describing and makes stale plans,
handoff packets, summaries, and reports fail validation if they omit the
operator-review acceptance state. It does not start or stop a backend, contact
WING, query console state, or grant live authority.

## Rejected alternatives

Keeping the field optional leaves a gap in the mandatory operator-review trail.
Adding ad hoc checks would duplicate the shared list contract and make future
staleness harder to diagnose.

## Implementation plan

- Extend the shared backend-port prestart failure-detail field list.
- Update WING MVP rails so the current first blocker requires the field.
- Add explicit rail tests for no-WING constants, Project Director output, and
  MVP docs.
- Regenerate latest MVP progress.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_supervised_wing_proof_plan.py tests/test_autonomy_promotion_readiness.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to stale artifact contract failures. Rollback removes the field
from the shared list and rail docs, then regenerates `MVP_PROGRESS_LATEST.md`.
No live WING behavior or readiness closure is introduced.
