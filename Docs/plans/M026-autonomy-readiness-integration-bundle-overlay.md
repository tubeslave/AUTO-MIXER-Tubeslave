# M026 Autonomy Readiness Integration Bundle Overlay

## Goal

Bind autonomy promotion readiness to the offline integration bundle contract
status before supervised or autonomous promotion planning can treat integration
evidence as usable planning input.

## Scope

- Offline/control-plane only.
- No WING contact.
- No backend contact.
- No OSC or websocket transport.
- No live write.
- No readiness closure, Full Autonomous eligibility, or live authority grant.

## Checkpoints

### M026-CP01 Readiness Overlay

Update `backend/autonomy_promotion_readiness.py` so
`build_autonomy_promotion_readiness_snapshot()` evaluates an optional
`offline_integration_bundle_manifest` through
`backend/evaluation_integration_bundle_contract.py`.

If the bundle is missing or fails the integration contract, the snapshot must
surface `offline_integration_bundle_contract_not_ready` as a global blocker.
The overlay must keep `live_authority_granted: false`,
`readiness_closure_allowed: false`, `live_write: false` and
`hardware_evidence: false`.

### M026-CP02 Project Rail and Evidence Sync

Update tests, roadmap, completion gate, capability matrix, current state and
autopilot evidence so M026 is durable and M001 remains
`WAITING_FOR_EXTERNAL_VALIDATION`.

## Acceptance Criteria

- Missing integration bundle evidence fails closed in autonomy readiness.
- Passing offline integration bundle evidence removes only the integration
  bundle blocker.
- offline integration bundle evidence can inform promotion planning only; it
  cannot grant authority or readiness.
- Offline integration bundle evidence can inform promotion planning only; it
  cannot grant live authority, close readiness, or mark Full Autonomous ready.
- Live/write/hardware claims inside the bundle are surfaced as source claims
  while the readiness snapshot remains read-only.
- M001 still waits for a real WING read-only artifact and operator review.

## Validation

```bash
PYTHONPATH=backend python -m pytest tests/test_autonomy_promotion_readiness.py -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m026_autonomy_readiness_consumes_integration_bundle_status -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Rollback

Revert the M026 overlay, tests and docs. Autonomy promotion readiness returns
to the previous state where offline integration bundle readiness is not yet
reported in the promotion snapshot.
