# Autopilot Heartbeat

Last heartbeat UTC: 2026-06-19T17:38:18Z

Status: WAITING_FOR_EXTERNAL_VALIDATION

Active milestone: M001

Active checkpoint: M001-EXTERNAL-WING-READONLY-ARTIFACT

Writer lease: none

Next action: wait for operator-supervised WING read-only preflight using `ops/autopilot/M001_CP04_wing_readonly_preflight_plan.json`.

Latest check: M015 replay threshold policy binding is now guarded by
`tests/test_offline_replay_evaluation_contract.py`,
`tests/test_evaluation_integration_bundle_contract.py` and
`tests/test_wing_mvp_rails.py::test_m015_replay_threshold_policy_binding_requires_m014_manifest`.
The updated `backend/offline_replay_evaluation_contract.py` contract consumes
the M014 `quality_threshold_policy_manifest` before replay promotion. Legacy
loose `threshold_policy` fields remain summary context only and cannot satisfy
approved-threshold requirements without the bound M014 contract manifest. It
keeps
`live_authority_granted: false`, `readiness_closure_allowed: false`,
`live_write: false` and `hardware_evidence: false`.
Targeted validation passed with `6 passed`, `5 passed` and `1 passed`; the full
rail file passed with `28 passed`; the full suite passed with
`4105 passed, 1 skipped, 1 warning`.
The fresh no-WING bundle remains
`artifacts/no_wing_release_check/20260619T135106Z_no_wing_release_check`;
the detached verifier passed with `failed_checks: []`, top-level coverage
JSON/Markdown checks true, `contacts_wing: false`, `contacts_backend: false`
and `live_authority_granted: false`. This heartbeat made no product/runtime
change to live console authority. M001 still has no real WING read-only artifact under
`artifacts/wing_readonly_preflight/`.
