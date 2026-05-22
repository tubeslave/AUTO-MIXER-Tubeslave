# Automixer Pre-Live Gain Review - 2026-05-21

## Verdict

Status: **pre-live review passed, physical WING gain proof not run**.

This review implements the next safe step after RC staging: verify the intended
supervised proposal path, record remaining write-surface boundaries, and refresh
the fake gain proof before any physical WING validation.

The current allowed hardware claim remains:

**Supervised hardware-proven for one fader proposal path.**

## Code Review Findings

- Current cockpit apply path uses `apply_operator_proposal`:
  `frontend/src/services/useOperatorBackend.ts` calls
  `websocketService.applyOperatorProposal(...)`.
- Main cockpit sources are covered by a unit guard that rejects direct calls to
  `.setFader(`, `.setGain(`, `.setEQ(`, `.setCompressor(`, `.routeOutput(`,
  `.loadSnap(`, and `.saveSnap(`.
- Backend operator mode policy allows live writes only in `Supervised`, and
  only for `fader` / `gain`.
- Backend Product Layer proposal apply blocks non-supervised modes with
  `operator_mode_blocks_live_write`, requires explicit approval, checks allowed
  value type, and routes apply through `_apply_manual_console_write`.
- WING deployment handlers route websocket `set_fader` / `set_gain` through the
  manual gate when the WING write boundary is active.
- EQ, compressor, routing, snapshot, FX, phase/delay and similar surfaces remain
  quarantined, fail-closed, or outside the approved cockpit path.

Important boundary:

- Direct methods still exist in the shared legacy websocket service and some
  legacy/non-cockpit runner surfaces. They are not the approved next proof path.
  The next physical validation must use
  `tools/wing_operator_proposal_validation_runner.py`, not the legacy direct
  smoke runner.

## Verification

Full handoff checks passed after this review:

- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`:
  `1126 passed, 1 skipped`.
- `npx tsc --noEmit` in `frontend/`: passed.
- `npm --prefix frontend test -- --watch=false`: `13 passed`.
- `npm --prefix frontend run test:e2e`: `8 passed`.
- `npm --prefix frontend run build`: passed.
- `npm --prefix frontend run dist:frontend`: passed.
- `git diff --check`: passed.

Package checks passed after the fresh `dist:frontend` build:

- Packaged backend server is executable.
- Required config resources are present.
- `app.asar` contains `/package.json`, `/public/electron.js`, and
  `/build/index.html`.
- Package resource scan found no `.env`, `*secret*`, `*token*`, or
  `*credential*` file names.
- App is Apple Development signed with hardened runtime; `spctl` still rejects
  it for public distribution because it is not a Developer ID notarized
  release artifact.

Focused safety and runner tests passed:

```bash
PYTHONPATH=backend python -m pytest \
  tests/test_wing_operator_proposal_validation_runner.py \
  tests/test_gain_proof_dry_run_runner.py \
  tests/test_server.py::TestWingDeploymentWriteBoundary -q
# 73 passed
```

Fake gain proof dry-run passed:

```bash
python3 tools/gain_proof_dry_run_runner.py \
  --channel 3 \
  --initial-gain-db 1.0 \
  --delta-db -0.25 \
  --cooldown-sec 2 \
  --artifact-root artifacts/wing_telemetry/gain_proof_dry_run
```

Latest fake run:

- Run id: `20260521T175356Z_gain_proof_dry_run_ch3`
- Summary:
  `artifacts/wing_telemetry/gain_proof_dry_run/20260521T175356Z_gain_proof_dry_run_ch3/summary.json`
- Physical WING: `false`
- Write policy: `fake_mixer_apply_operator_proposal_only`
- Apply readback: `verified_from_physical_readback` on fake mixer
- Rollback readback: `verified_from_physical_readback` on fake mixer
- Cooldown proof: blocked
- Emergency-stop proof: `emergency_stop_active`

## Next Physical WING Step

Do not run physical gain proof as an automatic continuation.

Allowed physical command shape remains:

```bash
python3 tools/wing_operator_proposal_validation_runner.py \
  --url ws://localhost:8765 \
  --channel <LOW_RISK_CHANNEL> \
  --value-type gain \
  --delta-db -0.25 \
  --timeout 8 \
  --rollback-delay-sec 1 \
  --execute \
  --prove-blocked-apply \
  --prove-emergency-stop
```

Before that command, the operator must confirm:

- WING is connected and read-only preflight is green.
- Backend is in `Supervised`.
- Emergency stop/reset-required state is explicitly cleared.
- Old gain value is physically/readably captured.
- The channel is low-risk and the move is a small gain trim-down.

Success wording is allowed only after physical apply readback, rollback,
rollback readback, cooldown/blocked proof, and artifact preservation:

**Supervised hardware-proven for fader and one gain trim-down path.**

If any proof point is missing, status remains `partial/blocked`.
