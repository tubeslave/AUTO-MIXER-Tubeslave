# ADR: WING MVP Backend Prestart Gate Authority Denial Summary JSON Surface

Date: 2026-06-15

## Context

The WING read-only handoff packet already carries backend prestart
authority-denial fields and packet-vs-plan match verdicts. The no-WING
`summary.json` exposed the gate object but still left raw denial fields nested,
while the Project Director current command requires those fields before any
backend/WING session.

## Options considered

1. Keep authority-denial fields nested inside the handoff artifact.
2. Add a dedicated top-level authority-denial summary surface.
3. Add a broad handoff-artifact projection layer.

## Decision

Add a dedicated top-level authority-denial summary surface to the existing
backend prestart gate summary helper.

## Why this won

The dedicated surface is small, mirrors existing validated handoff values, and
makes the current first-command no-authority contract machine-checkable without
changing live WING authority.

## Rejected alternatives

Nested-only evidence keeps the audit path brittle. A generic projection layer is
larger than this specific backend prestart authority-denial gap requires.

## Implementation plan

- Extend `build_backend_port_prestart_gate_summary_surface()` with
  authority-denial and match fields.
- Assert top-level values match the nested handoff artifact in the focused
  no-WING summary test.
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

Risk: readers could confuse the exposed no-authority fields with authorization.
Mitigation: all projected authority values remain `false`, match verdicts remain
auditable, and the surface explicitly denies readiness closure and live
authority. Rollback is to remove the projection, test assertions, and Council
evidence.
