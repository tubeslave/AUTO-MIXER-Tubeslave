# ADR: WING MVP Backend Prestart Gate Object Summary JSON Surface

Date: 2026-06-15

## Context

The WING read-only handoff packet already carries a structured
`backend_port_prestart_gate` object and expected gate object. The no-WING
`summary.json` exposed match verdicts but did not expose the objects at the top
level, forcing machine consumers to parse the nested handoff artifact to verify
the first operator-step gate.

## Options considered

1. Keep only top-level match verdicts.
2. Add top-level backend prestart gate objects.
3. Add a generic handoff-artifact projection framework.

## Decision

Add top-level backend prestart gate objects to the existing backend prestart
gate summary surface.

## Why this won

The change is narrow, follows the existing summary-surface helper pattern, and
makes the Project Director current command directly checkable without changing
WING proof authority.

## Rejected alternatives

Keeping only verdicts leaves the exact object contract hidden. A generic
projection framework is larger than this specific first-step gate gap requires.

## Implementation plan

- Extend `build_backend_port_prestart_gate_summary_surface()` with the gate,
  expected gate, and declared expected gate objects.
- Assert the top-level objects match the nested handoff artifact in the focused
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

Risk: the top-level object could be read as authority to start a backend/WING
session. Mitigation: the gate object and summary-surface flags continue to deny
live authority. Rollback is to remove the object projection, test assertions,
and Council evidence.
