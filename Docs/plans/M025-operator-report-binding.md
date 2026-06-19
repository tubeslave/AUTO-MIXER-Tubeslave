# M025 Operator Report Binding

## Goal

Bind offline integration promotion to the M024 operator-report contract result
instead of accepting loose `operator_report_bound` integration checks as enough
evidence.

## Scope

- Offline/control-plane only.
- No WING contact.
- No backend contact.
- No OSC or websocket transport.
- No live write.
- No readiness closure or live authority grant.

## Checkpoints

### M025-CP01 Integration Bundle Binding

Update `backend/evaluation_integration_bundle_contract.py` so integration
bundle evaluation consumes `operator_report_manifest`, evaluates it through
`backend/operator_report_contract.py`, and reports a fail-closed
`operator_report_contract_not_ready` blocker when the child contract is not
ready.

Loose `operator_report_bound` integration checks remain visible as bundle
context, but they cannot satisfy the operator-report requirement without the
bound M024 contract manifest.

### M025-CP02 Project Rail and Evidence Sync

Update tests, roadmap, completion gate, capability matrix, current state and
autopilot evidence so the rail makes the dependency explicit and keeps M001 in
`WAITING_FOR_EXTERNAL_VALIDATION`.

## Acceptance Criteria

- Complete offline integration bundles include a passing
  `operator_report_manifest`.
- Bundles with only `operator_report_bound` and no M024 manifest fail closed.
- Operator-report child blockers are surfaced with the `operator_report:`
  prefix.
- The integration bundle still always returns `live_authority_granted: false`.
- M001 still waits for a real WING read-only artifact and operator review.

## Validation

```bash
PYTHONPATH=backend python -m pytest tests/test_evaluation_integration_bundle_contract.py -q
PYTHONPATH=backend python -m pytest tests/test_operator_report_contract.py -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m025_operator_report_binding_requires_m024_manifest -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Rollback

Revert the M025 code, tests and docs changes. That returns integration bundles
to the previous M024 state where the operator-report evaluator exists but is
not yet required by integration promotion.
