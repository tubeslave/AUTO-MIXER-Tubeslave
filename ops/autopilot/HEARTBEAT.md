# Autopilot Heartbeat

Last heartbeat UTC: 2026-06-19T16:56:13Z

Status: WAITING_FOR_EXTERNAL_VALIDATION

Active milestone: M001

Active checkpoint: M001-EXTERNAL-WING-READONLY-ARTIFACT

Writer lease: none

Next action: wait for operator-supervised WING read-only preflight using `ops/autopilot/M001_CP04_wing_readonly_preflight_plan.json`.

Latest check: M012 fault-injection recovery evaluation is now guarded by
`tests/test_fault_injection_recovery_contract.py` and
`tests/test_wing_mvp_rails.py::test_m012_fault_injection_recovery_contract_stays_simulator_only_and_audit_gated`.
The new `backend/fault_injection_recovery_contract.py` contract requires
transport-timeout, connection-loss, stale-state, emergency-stop,
safety-kernel-reject and rollback-required scenarios, simulator /
fault-injection / recovery evidence, recovery outcomes, audit fields and
approved recovery policy before simulator fault-recovery promotion. Even when
fault recovery promotion is ready, it keeps `live_authority_granted: false`,
`readiness_closure_allowed: false`, `live_write: false` and
`hardware_evidence: false`.
Targeted validation passed with `5 passed` and `1 passed`; the full rail file
passed with `25 passed`; the full suite passed with
`4091 passed, 1 skipped, 1 warning`.
The fresh no-WING bundle remains
`artifacts/no_wing_release_check/20260619T135106Z_no_wing_release_check`;
the detached verifier passed with `failed_checks: []`, top-level coverage
JSON/Markdown checks true, `contacts_wing: false`, `contacts_backend: false`
and `live_authority_granted: false`. This heartbeat made no product/runtime
change. M001 still has no real WING read-only artifact under
`artifacts/wing_readonly_preflight/`.
