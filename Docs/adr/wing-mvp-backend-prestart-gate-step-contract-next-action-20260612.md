# ADR: WING MVP Backend Prestart Gate Step Contract Next Action

Date: 2026-06-12

## 1. Context

The no-WING handoff packet validates `backend_port_prestart_gate` against
`expected_backend_port_prestart_gate`. The structured gate includes the first
operator-step contract, but the WING MVP selected action did not name those
structured fields explicitly.

## 2. Options Considered

- Keep only the object equality requirement.
- Add the structured gate first-step fields to the selected action and autoloop
  instructions.

## 3. Decision

Add `backend_port_prestart_gate.operator_sequence_step_index: 0` and
`backend_port_prestart_gate.operator_sequence_step_must_be_first: true` to the
operator-facing next-action surfaces.

## 4. Why This Won

The structured gate is the contract reviewers inspect before any supervised
fader/gain proof. Naming the first-step fields makes the review requirement
explicit without widening live authority.

## 5. Rejected Alternatives

Relying only on full-object equality was rejected because it hides the first-step
requirement inside a larger structure.

## 6. Implementation Plan

- Update the blocker selected action and current first command.
- Update autoloop prompt and README.
- Update the MVP stop/current-action text.
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
