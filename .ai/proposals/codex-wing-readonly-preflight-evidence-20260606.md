# Codex Proposal: WING Read-Only Preflight Evidence

## Short thesis

Add a fail-closed evidence matrix for read-only WING preflight artifacts so a
real hardware preflight run can feed readiness review without widening live
authority.

## What I understood

The active priority is the supervised WING Rack MVP. The previous runner records
read-only backend state and physical readbacks, but its `summary.json` needs a
transport-free consumer before it can safely inform readiness.

## Proposed solution

- Create `backend/wing_readonly_preflight_evidence.py`.
- Read only existing `summary.json` files or inline payloads.
- Accept only `wing_readonly_preflight.v1` artifacts that:
  - passed with `connection_mode=wing`;
  - required physical readback;
  - have all readbacks physically verified;
  - sent only `get_operator_mode`, `get_state`, and `query_mixer_parameter`;
  - sent no write, proposal, rollback, emergency, connect, or disconnect
    requests.
- Emit an MVP blocker matrix for read-only connection, physical readback,
  operator evidence, and readiness matrix update candidates.
- Keep readiness closure and live authority false until manual operator review.
- Add the evaluator to the no-WING release check as a schema gate and summary
  section, still without allowing readiness closure or live authority.

## Likely files to touch

- `backend/wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/INDEX.md`

## Alternatives considered

- Feed preflight summaries into `supervised_proof_artifact_evaluator`. Rejected:
  that evaluator correctly requires apply, rollback, emergency-stop, and
  blocked-followup proof; preflight must not become supervised apply evidence.
- Leave the runner artifact standalone. Rejected: standalone JSON is easy to
  misread and does not create a fail-closed readiness view.

## Risks

The matrix can be mistaken for closed proof. The evaluator therefore reports
candidate rows only, keeps `closed=false`, and requires manual operator review.

## Test plan

- `python3 -m py_compile tools/no_wing_release_check.py backend/wing_readonly_preflight_evidence.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_processing_registry.py::test_registry_snapshot_can_hide_full_method_inventory tests/test_project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may prefer updating the existing supervised proof report instead. I would
keep preflight evidence separate because it is not apply proof and should not
share the promotion-candidate path.
