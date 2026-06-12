# ADR: WING MVP Supervised Gate Required Markers Next Action

Date: 2026-06-12

## 1. Context

The no-WING handoff packet validates and summarizes
`required_supervised_prestart_gate_markers`. The WING MVP selected action named
the marker strings in the final operator step, but did not explicitly require
the structured marker-list field.

## 2. Options Considered

- Keep only the final operator step marker wording.
- Add the structured marker-list field to selected-action and autoloop surfaces.

## 3. Decision

Add `required_supervised_prestart_gate_markers` to the operator-facing
next-action surfaces, with the expected markers:
`accepted_decision_ready`, `valid accepted-candidate prestart trace`, and
`WING_SUPERVISED_FADER_GAIN_PROOF`.

## 4. Why This Won

The structured marker list is easier for reviewers and automated cycles to check
than a sentence inside the final operator step. It does not grant live authority.

## 5. Rejected Alternatives

Relying only on the final step text was rejected because it hides the
machine-readable marker contract.

## 6. Implementation Plan

- Update the blocker selected action and current first command.
- Update the read-only runbook and MVP doc.
- Update autoloop prompt and README.
- Refresh generated MVP progress.
- Add tests on the affected text surfaces.

## 7. Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Risks And Rollback

Risk is limited to documentation/test wording drift. Rollback by reverting this
ADR and the related text/test changes; no live mixer behavior is changed.
