# M027 Operator Adapter Integration Bundle Readiness Summary

## Goal

Expose the M026 offline integration bundle readiness overlay in the
operator-facing module status adapter so the UI can show the integration bundle
status, blockers, missing checks and authority-denial state without granting
runtime authority.

## Scope

- Frontend adapter contract only.
- Offline/control-plane only.
- No WING contact.
- No backend contact.
- No OSC or websocket transport.
- No live write.
- No readiness closure, Full Autonomous eligibility, or live authority grant.

## Checkpoints

### M027-CP01 Operator Adapter Summary

Update `frontend/src/services/operatorAdapters.ts` so the existing
`autonomy_promotion_summary` module status row consumes
`summary.offline_integration_bundle` from `autonomy_promotion_readiness`.

The row must surface:

- `integration_bundle_ready:*`;
- `integration_bundle_status:*`;
- `integration_bundle_source_authority_denied:*`;
- `integration_bundle_blocker:*`;
- `integration_bundle_missing_check:*`;
- child contract status tags.

### M027-CP02 Contract Test and Evidence Sync

Update `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`,
roadmap, completion gate, capability matrix, current state and autopilot
evidence so M027 is durable and M001 remains
`WAITING_FOR_EXTERNAL_VALIDATION`.

## Acceptance Criteria

- The operator adapter surfaces
  `offline_integration_bundle_contract_not_ready` as a read-only blocker.
- The adapter reports integration bundle readiness and status in
  `pipelineProfiles`.
- The adapter keeps `canWriteLive: false` and `liveWriteKinds: []`.
- Passing or failing frontend summary state cannot grant live authority,
  readiness closure, hardware evidence or Full Autonomous readiness.
- M001 still waits for a real WING read-only artifact and operator review.

## Validation

```bash
npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m027_operator_adapter_surfaces_integration_bundle_readiness_summary -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Rollback

Revert the M027 frontend adapter/test changes and the accompanying
docs/state/evidence. The backend M026 readiness overlay remains intact, but the
operator-facing adapter stops rendering integration bundle summary tags.
