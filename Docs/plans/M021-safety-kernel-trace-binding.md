# Plan M021: Safety kernel trace binding

## Checkpoints

| Checkpoint | Scope | Files | Evidence requirement | Safety boundary | Validation | Exit condition | Rollback | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M021-CP01 | Bind offline integration promotion to the safety-kernel trace contract | `backend/evaluation_integration_bundle_contract.py`, `tests/test_evaluation_integration_bundle_contract.py` | Integration bundle manifests must include a ready M020 `safety_kernel_trace_manifest`; loose `safety_kernel_trace_bound` checks are not sufficient | Offline integration evidence only, no hardware, no OSC writes | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q` | legacy trace-bound-only bundles remain blocked; bound manifest passes only offline | revert M021 commit | completed |
| M021-CP02 | Bind M021 to roadmap, completion gates and autopilot state | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/PROJECT_COMPLETION_GATE.md`, `Docs/adr/safety-kernel-trace-binding-20260619.md`, `tests/test_wing_mvp_rails.py` | M021 remains pending until integration bundle evidence uses the safety-kernel trace contract result | Report-only gate binding | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m021_safety_kernel_trace_binding_requires_m020_manifest -q` | roadmap/capability/current-state/report rows | revert M021 commit | completed |

## Validation Contract

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_safety_kernel_trace_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m021_safety_kernel_trace_binding_requires_m020_manifest -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Acceptance Criteria

- Integration bundle manifests consume `safety_kernel_trace_manifest` from M020
  before reporting offline integration promotion readiness.
- loose `safety_kernel_trace_bound` integration checks without the M020
  contract manifest remain visible as integration-check context but cannot
  satisfy safety-kernel trace requirements.
- Safety-kernel trace contract blockers are surfaced with the
  `safety_kernel_trace:` child prefix.
- Bound safety-kernel trace manifests still cannot grant live authority, close
  readiness, contact WING/backend, send live writes or become hardware proof.
