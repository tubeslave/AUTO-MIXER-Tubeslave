# M028 Main Pipeline Preview Integration Bundle Readiness Summary

## Goal

Expose the M026 offline integration bundle readiness overlay in
`main_pipeline_preview.summary` so preview, Director and operator-runtime
consumers can read integration bundle status without walking the nested
registry snapshot.

## Scope

- Backend preview summary only.
- Offline/control-plane only.
- No WING contact.
- No backend contact beyond local pure snapshot construction.
- No OSC or websocket transport.
- No live write.
- No queue mutation.
- No readiness closure, Full Autonomous eligibility, or live authority grant.

## Checkpoints

### M028-CP01 Preview Summary Overlay

Update `backend/processing_pipeline_preview.py` so the existing read-only
`main_pipeline_preview.summary` copies the following fields from
`registry.autonomy_promotion_readiness.summary.offline_integration_bundle`:

- readiness boolean;
- status;
- blockers;
- missing integration checks;
- source authority denial;
- live authority, readiness closure, live write and hardware-evidence flags.

### M028-CP02 Contract Test and Evidence Sync

Add an isolated preview-summary regression test, update roadmap, completion
gate, capability matrix, current state and autopilot evidence so M028 is
durable and M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`.

## Acceptance Criteria

- `main_pipeline_preview.summary` surfaces offline integration bundle readiness.
- Blocked or missing bundle evidence remains visible as read-only blocker data.
- Summary fields match the nested autonomy promotion readiness overlay.
- Preview stays read-only: no queue mutation, no live write, no mixer calls.
- Passing preview summary state cannot grant live authority, readiness closure,
  hardware evidence or Full Autonomous readiness.
- M001 still waits for a real WING read-only artifact and operator review.

## Validation

```bash
PYTHONPATH=backend python -m pytest tests/test_processing_pipeline_preview_integration_bundle_summary.py -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m028_main_pipeline_preview_surfaces_integration_bundle_readiness_summary -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Rollback

Revert the M028 preview summary fields, isolated test and accompanying
docs/state/evidence. The backend M026 readiness overlay and M027 frontend
adapter summary remain intact.
