# Plan M020: Safety kernel trace contract

## Checkpoints

| Checkpoint | Scope | Files | Evidence requirement | Safety boundary | Validation | Exit condition | Rollback | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M020-CP01 | Add fail-closed safety-kernel trace contract | `backend/safety_kernel_trace_contract.py`, `tests/test_safety_kernel_trace_contract.py` | Offline traces must cover deterministic allow/reject scenarios, guard checks and audit fields before integration use | Offline trace evidence only, no hardware, no OSC writes | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_safety_kernel_trace_contract.py -q` | trace result keeps `live_authority_granted: false` and `readiness_closure_allowed: false` | revert M020 commit | completed |
| M020-CP02 | Bind M020 to roadmap and completion gates | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/PROJECT_COMPLETION_GATE.md`, `Docs/adr/safety-kernel-trace-contract-20260619.md`, `tests/test_wing_mvp_rails.py` | M020 remains pending until deterministic safety-kernel trace evidence is complete and offline-only | Report-only gate binding | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m020_safety_kernel_trace_contract_stays_offline_and_non_bypassable -q` | roadmap/capability/current-state/report rows | revert M020 commit | completed |

## Validation Contract

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_safety_kernel_trace_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m020_safety_kernel_trace_contract_stays_offline_and_non_bypassable -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Acceptance Criteria

- Safety-kernel trace manifests require deterministic allow/reject scenarios,
  guard checks and audit fields.
- Missing scenarios, missing guard checks, missing audit fields and wrong
  safety decisions fail closed.
- Models, LLMs, classifiers, memory and proposal queues cannot use this trace
  evidence to bypass deterministic safety checks.
- Safety-kernel trace evidence cannot grant live authority, close readiness,
  contact WING/backend, send live writes or become hardware proof.
