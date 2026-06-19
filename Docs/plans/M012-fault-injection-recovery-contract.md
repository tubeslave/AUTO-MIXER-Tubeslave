# Plan M012: Fault-injection recovery contract

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M012-CP01 | Add fail-closed simulator fault recovery contract | `backend/fault_injection_recovery_contract.py`, `tests/test_fault_injection_recovery_contract.py` | Incomplete transport-timeout, connection-loss, stale-state, emergency-stop, safety-kernel and rollback bundles are blocked | Offline simulator only, no hardware, no OSC writes | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_fault_injection_recovery_contract.py -q` | contract result keeps `live_authority_granted: false` and `readiness_closure_allowed: false` | revert M012 commit | completed |
| M012-CP02 | Bind recovery contract to roadmap and completion gates | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/CONSOLE_CAPABILITY_MATRIX.md`, `Docs/adr/fault-injection-recovery-contract-20260619.md`, `tests/test_wing_mvp_rails.py` | M012 remains pending until required fault scenarios, recovery outcomes, approved policy and audit evidence exist | Report-only gate binding | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m012_fault_injection_recovery_contract_stays_simulator_only_and_audit_gated -q` | roadmap/capability/current-state/report rows | revert M012 commit | completed |

## Required Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_fault_injection_recovery_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Acceptance Criteria

- Simulator fault evidence cannot grant live authority or readiness closure.
- Simulator fault evidence cannot be labeled as hardware evidence.
- Transport-timeout, connection-loss, stale-state, emergency-stop,
  safety-kernel-reject and rollback-required scenarios are all required.
- Simulator, fault-injection and recovery evidence are all required per fault
  scenario.
- Fault detection, fail-closed behavior, unsafe-action blocking, deterministic
  safety-kernel enforcement, rollback verification, emergency-stop verification,
  operator visibility, audit logging and no-live-write outcomes are all
  required.
- Approved recovery policy requires explicit owner and decision id.
- Audit records must contain fault id, trigger, detection time, decision id,
  operator message and rollback reference.

## Rollback

Revert the M012 commit. The contract is not wired into live runtime paths and
has no console transport, so rollback is a code/docs/test revert only.
