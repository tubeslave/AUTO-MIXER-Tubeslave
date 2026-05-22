# Automixer Handoff + WING Supervised Validation - 2026-05-20

## Verdict

Status: **supervised hardware-proven for one narrow fader proposal path**.

This is not broad autonomous readiness. The proven live path is limited to one
WING channel fader change through Product Layer proposal apply:

- `create_operator_proposal`
- `apply_operator_proposal`
- `rollback_last_manual_write`
- readback after apply and rollback
- emergency stop blocks a later proposal apply

No EQ, phase, delay, routing, snapshots, scenes, training writes, rule writes,
or autonomous apply were tested or enabled.

## Repo And Package State

The worktree is intentionally dirty. It contains older frontend/backend/package
work plus this WING validation slice. Do not treat the whole worktree as one
clean commit without staging review.

Package artifacts verified:

- macOS app: `frontend/dist/mac-arm64/AUTO MIXER Tubeslave.app`
- DMG: `frontend/dist/AUTO MIXER Tubeslave-1.0.0-arm64.dmg`
- Packaged backend executable:
  `frontend/dist/mac-arm64/AUTO MIXER Tubeslave.app/Contents/Resources/backend/server`
- Source backend executable: `backend/dist/server`
- Frontend in asar: `/build/index.html`, `/public/electron.js`, `/public/preload.js`
- Config resources:
  - `config/default_config.json`
  - `config/automixer.yaml`
  - `config/muq_eval.yaml`
  - `config/perceptual.yaml`
  - `config/source_knowledge.yaml`

Secret/package check:

- No `.env`, `*.env`, `*token*`, or `*secret*` files were found under packaged
  `Contents/Resources` for arm64 or x64 app outputs.

Screenshots:

- 16-route backend-connected screenshots:
  `Docs/reports/automixer-ui-screens-20260519/backend-connected/`
- Additional dashboard screenshots:
  `Docs/reports/automixer-dashboard-screens-20260520/`

## Verification Commands

Passed:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
# 1075 passed, 1 skipped

PYTHONPATH=backend python -m pytest tests/test_wing_operator_proposal_validation_runner.py -q
# 9 passed

npm --prefix frontend run test:e2e
# 8 passed

npm --prefix frontend run build
# passed

npm --prefix frontend run dist
# passed, built x64 and arm64 Electron packages

git diff --check
# passed
```

Package checks passed:

```bash
test -x "frontend/dist/mac-arm64/AUTO MIXER Tubeslave.app/Contents/Resources/backend/server"
test -f "frontend/dist/mac-arm64/AUTO MIXER Tubeslave.app/Contents/Resources/config/default_config.json"
npx --prefix frontend asar list "frontend/dist/mac-arm64/AUTO MIXER Tubeslave.app/Contents/Resources/app.asar"
```

Note: the live validation used the already running backend process on
`ws://localhost:8765` (`backend/server.py`, PID 23699). That process was not
killed or restarted.

## WING Preflight

Read-only checks showed:

- Backend WebSocket: `ws://localhost:8765`
- WING connected: `192.168.1.102`
- Connection mode: `wing`
- Client: `EnhancedOSCClient`
- Test target: channel 16
- Channel name: `VOCAL`
- Initial fader: `0.0 dB`
- Planned fader target: `-0.25 dB`
- Runner policy: `apply_operator_proposal_only`

Dry-run artifact:

- `artifacts/wing_telemetry/20260520T124838Z_wing_operator_proposal_ch16_fader/summary.json`

## Live Validation

Runner:

```bash
python3 tools/wing_operator_proposal_validation_runner.py \
  --url ws://localhost:8765 \
  --channel 16 \
  --value-type fader \
  --delta-db -0.25 \
  --timeout 8 \
  --rollback-delay-sec 1 \
  --execute \
  --prove-blocked-apply \
  --prove-emergency-stop
```

Successful run:

- Run id: `20260520T125057Z_wing_operator_proposal_ch16_fader`
- Summary:
  `artifacts/wing_telemetry/20260520T125057Z_wing_operator_proposal_ch16_fader/summary.json`
- Events:
  `artifacts/wing_telemetry/20260520T125057Z_wing_operator_proposal_ch16_fader/events.jsonl`
- Manual write gate events:
  `artifacts/wing_telemetry/manual_write_gate/manual_write_events.jsonl`

Proof points:

- Assist mode blocked apply with reason `operator_mode_blocks_live_write`.
- Supervised mode allowed apply only through `apply_operator_proposal`.
- Direct `set_fader` / `set_gain` messages were not sent by the runner.
- Apply changed CH16 fader `0.0 -> -0.25`.
- Apply readback: `verified_from_cache`, value `-0.25`.
- Rollback restored CH16 fader `-0.25 -> 0.0`.
- Rollback readback: `verified_from_cache`, value `0.0`.
- Emergency stop then blocked a later proposal apply with reason
  `emergency_stop_active`.
- Final CH16 fader readback after validation: `0.0`.

Important operational note:

- Backend emergency stop is now active by design from the proof step.
- `dashboard_snapshot.safety.manual_write_supervision.emergency_stopped == true`.
- `ready_for_live == false` while emergency stop or post-emergency automation
  freeze is active.
- The code now exposes `reset_manual_write_emergency_stop`, but it requires
  explicit operator approval, writes an audit event, and does not silently
  unfreeze automation. Do not run it as an automatic recovery step.

## First Failed Attempt

Before adding a rollback delay, run
`20260520T124930Z_wing_operator_proposal_ch16_fader` partially succeeded:

- Apply via `apply_operator_proposal`: passed.
- Readback after apply: passed.
- Immediate rollback: failed with `mixer_write_failed`.
- CH16 remained at `-0.25`.

Recovery was done through a second Product Layer proposal, not direct OSC:

- Proposal id: `restore-ch16-fader-20260520T1250Z`
- Restore target: CH16 fader `-0.25 -> 0.0`
- Readback: `verified_from_cache`

Runner hardening added after this:

- `--rollback-delay-sec`
- blocked-apply proof now explicitly switches to `Assist` before attempting the
  negative control
- test proposals are dismissed after blocked proof

## Next Work

1. Keep the current status wording narrow: **supervised hardware-proven fader
   proposal path**, not autonomous ready.
2. Before any further live write, use the explicit operator-approved emergency
   reset/status flow and verify automation freeze/readiness state.
3. Finish Phase 1 write-path inventory and keep unsupported surfaces fail-closed.
4. Extend the same proposal-gated proof to gain only after a fresh old-value
   pre-read and rollback evidence are available.
5. Continue improving proposal builders and cockpit operator workflow without
   enabling autonomous WING writes.
