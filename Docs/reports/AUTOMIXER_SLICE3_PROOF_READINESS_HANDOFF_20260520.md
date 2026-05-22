# AUTOMIXER Slice 3 Proof-Readiness Handoff - 2026-05-20

## Scope

Slice 3 prepares the existing supervised `gain` path for a later physical WING
proof. It does not approve or run live WING/OSC commands.

## Staging-Ready Slice Files

Slice 1-2 registry, Product Layer, preview, and write-boundary files:

- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/processing_pipeline_preview.py`
- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `Docs/PROCESSING_METHODS_INVENTORY.md`
- `Docs/adr/processing-pipeline-selection.md`
- `Docs/reports/AUTOMIXER_PHASE1_WRITE_PATH_INVENTORY_20260520.md`
- `tests/test_processing_registry.py`
- `tests/test_operator_product_state.py`

Slice 3 proof-readiness files:

- `backend/server.py`
- `backend/wing_client.py`
- `backend/handlers/product_state_handlers.py`
- `tools/wing_operator_proposal_validation_runner.py`
- `tools/gain_proof_dry_run_runner.py`
- `tests/test_server.py`
- `tests/test_gain_proof_dry_run_runner.py`
- `frontend/src/services/__tests__/websocketCore.test.mjs`
- `Docs/PROJECT_IMPLEMENTATION_PLAN.md`
- `Docs/reports/AUTOMIXER_SLICE3_PROOF_READINESS_HANDOFF_20260520.md`

Do not include unrelated dirty files, frontend modernization files, generated
test screenshots/results, or legacy unfinished changes in this slice.

## Proof Contract

Manual supervised `fader`/`gain` writes now report:

- `pre_read_status`
- `pre_read_source`
- `readback_source`
- `rollback_readback_status`
- `rollback_proof_status`
- `proof_status`: `blocked`, `partial`, or `verified`

Readback status vocabulary:

- `verified_from_cache`
- `verified_from_physical_readback`
- `readback_unavailable`
- `not_verified`
- `mismatch_from_cache`
- `mismatch_from_physical_readback`

If pre-read is unavailable, the write fails closed with
`reason=pre_read_unavailable` and `proof_status=blocked`.

If apply or rollback writes happen but readback is not verified, the result is
`proof_status=partial`; it must not be treated as ready for physical proof.

Rollback requires the same evidence level as apply. A rollback write with
unverified readback is marked `rollback_status=partial`, not `applied`.

## Backend Boundary

Approved supervised path remains unchanged:

1. Product Layer proposal is created.
2. Operator mode is `supervised`.
3. Operator approves the apply.
4. `apply_operator_proposal` routes to `_apply_manual_console_write`.
5. Only `fader` and `gain` are allowed.
6. Rollback uses `rollback_last_manual_write`.

No new live endpoints were added.

`WingClient.query_parameter(address)` is read-only and sends an OSC query
without values. It exists only to let the manual gate distinguish cached
evidence from physical readback during a future operator-approved proof.

## Frontend Boundary

Cockpit source tests now assert that main cockpit flows do not call:

- `setFader`
- `setGain`
- `setEQ`
- `setCompressor`
- `routeOutput`
- `loadSnap`
- `saveSnap`

The expected cockpit apply path remains `applyOperatorProposal`.

Legacy service methods may remain available as service-level compatibility
surfaces, but the main cockpit must not use them as direct live buttons.

## Dry-Run Gain Proof Runner

`tools/gain_proof_dry_run_runner.py` runs one fake gain proposal through the
real Product Layer dispatch path:

- create proposal;
- apply proposal through supervised gate;
- physical-readback-shaped fake pre-read and readback;
- cooldown blocked proof;
- rollback readback proof;
- emergency-stop blocked proof;
- JSONL events and JSON summary artifacts.

Default behavior is fake/in-process only. It does not connect to WING, open OSC,
or start a backend WebSocket.

Physical gain proof is still a separate operator decision.

## Verification

Focused checks run during this slice:

- `PYTHONPATH=backend python3 -m py_compile backend/server.py backend/wing_client.py backend/handlers/product_state_handlers.py tools/wing_operator_proposal_validation_runner.py tools/gain_proof_dry_run_runner.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_server.py::TestWingDeploymentWriteBoundary tests/test_gain_proof_dry_run_runner.py tests/test_wing_operator_proposal_validation_runner.py tests/test_operator_product_state.py -q`
- `cd frontend && npm test`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `cd frontend && npm run build`
- `python3 tools/gain_proof_dry_run_runner.py --artifact-root /tmp/automixer_gain_proof_dry_run_check --channel 2 --delta-db -0.25 --cooldown-sec 1`
- `git diff --check`
- conflict-marker scan
- tracked-secret scan

Current result:

- backend focused: `80 passed`
- backend full: `1098 passed, 1 skipped`
- frontend tests: `13 passed`
- frontend build: passed
- standalone fake gain proof runner: passed
- `git diff --check`: passed
- conflict-marker scan: no matches
- tracked-secret scan: only existing dummy secret fixture in
  `tests/test_project_vector_index.py`; clean after excluding that fixture.

No live WING/OSC command was run for this slice.
