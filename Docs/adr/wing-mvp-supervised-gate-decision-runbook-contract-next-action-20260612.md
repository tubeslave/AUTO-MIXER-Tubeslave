# ADR: WING MVP Supervised Gate Decision Runbook Contract Next Action

Date: 2026-06-12

## 1. Context

The no-WING handoff packet validates `supervised_prestart_gate` against
`expected_supervised_prestart_gate`. The structured gate includes the required
operator decision status and the only runbook it unlocks, but the WING MVP
selected action did not name those structured fields explicitly.

## 2. Options Considered

- Keep only the object equality requirement.
- Add the structured accepted-decision and runbook fields to selected-action and
  autoloop surfaces.

## 3. Decision

Add `supervised_prestart_gate.requires_operator_decision_status:
accepted_decision_ready` and `supervised_prestart_gate.unlocks_runbook:
Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` to the operator-facing
next-action surfaces.

## 4. Why This Won

The structured gate is the contract reviewers inspect before any supervised
fader/gain proof. Naming these fields makes the review requirement explicit
without granting live authority.

## 5. Rejected Alternatives

Relying only on full-object equality was rejected because it hides the
accepted-decision and runbook boundary inside a larger structure.

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
