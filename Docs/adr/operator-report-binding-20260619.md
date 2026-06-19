# ADR: Operator Report Binding

## Context

M024 added a standalone operator-report contract for offline, operator-visible
review evidence. M013 integration bundle evaluation still accepted loose
`operator_report_bound` integration checks as context, which could imply that
the report was bound without proving the full M024 contract.

M001 remains blocked on an external real WING read-only artifact and operator
review. This ADR covers only offline integration evidence gating.

## Options Considered

1. Leave `operator_report_bound` as the only integration bundle signal.
2. Require `operator_report_manifest` and evaluate it with the M024 contract.
3. Inline the operator-report checks directly in the integration bundle.

## Decision

Require integration bundle manifests to include `operator_report_manifest` and
evaluate that manifest through `backend/operator_report_contract.py`.

The loose `operator_report_bound` check remains listed in
`REQUIRED_INTEGRATION_CHECKS` as integration context, but it cannot satisfy the
operator-report requirement without the bound M024 manifest.

The integration bundle always returns `live_authority_granted: false`.

## Why This Won

The M024 contract is already the single place that defines operator-visible
sections, child-result bindings, explicit authority-denial flags, visible
blockers, next operator action and audit fields. Reusing it avoids duplicated
logic and keeps future report requirements centralized.

## Rejected Alternatives

Leaving the loose check as sufficient was rejected because it can hide an empty
or incomplete operator report behind a passed integration flag.

Inlining the checks was rejected because it would split ownership between two
contracts and make drift likely.

## Implementation Plan

- Add an operator-report child result to
  `backend/evaluation_integration_bundle_contract.py`.
- Surface `operator_report_contract_not_ready` plus prefixed child blockers.
- Add the legacy blocker
  `legacy_operator_report_bound_without_contract_manifest` when
  `operator_report_bound` passes but no manifest is present.
- Update tests, roadmap, completion gate, capability matrix, current state and
  autopilot evidence.

## Test Plan

- `PYTHONPATH=backend python -m pytest tests/test_evaluation_integration_bundle_contract.py -q`
- `PYTHONPATH=backend python -m pytest tests/test_operator_report_contract.py -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m025_operator_report_binding_requires_m024_manifest -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: existing generated integration manifests without `operator_report_manifest`
will now fail closed. That is intentional for promotion gating.

Rollback: revert the M025 binding changes while keeping the standalone M024
operator-report evaluator available.
