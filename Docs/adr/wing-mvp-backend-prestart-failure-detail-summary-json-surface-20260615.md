# ADR: WING MVP Backend Prestart Failure Detail Summary JSON Surface

Date: 2026-06-15

## Context

The WING read-only handoff packet already carries the canonical backend
prestart failure-detail field list. The aggregate no-WING `summary.json` did
not expose that list at the top level, while the Project Director current
command requires it before any backend/WING session.

## Options considered

1. Keep failure-detail fields nested inside the handoff artifact.
2. Add a dedicated top-level failure-detail summary surface.
3. Add a broad handoff-artifact projection layer.

## Decision

Add a dedicated top-level backend prestart failure-detail summary surface.

## Why this won

The dedicated surface is small, mirrors the existing canonical contract, and
makes the current first-command fail-closed listener-review contract directly
auditable without changing live WING authority.

## Rejected alternatives

Nested-only evidence keeps the aggregate summary incomplete for automation.
A broad projection layer is larger than this current reporting gap requires.

## Implementation plan

- Extend `build_backend_port_prestart_gate_summary_surface()` with the
  failure-detail field list, expected list, and match verdict.
- Assert the top-level values match the nested handoff artifact and canonical
  contract in the focused no-WING summary test.
- Add Council report and index the evidence.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_wing_preflight_handoff_packet_artifact`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: readers could mistake a complete failure-detail surface for readiness.
Mitigation: the surface explicitly denies readiness closure and live authority.
Rollback is to remove the projection, test assertions, and Council evidence.
