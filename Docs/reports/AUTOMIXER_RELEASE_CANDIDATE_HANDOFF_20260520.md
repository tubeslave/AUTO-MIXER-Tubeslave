# Automixer Release-Candidate Handoff - 2026-05-20

## Verdict

Status: **MVP cockpit candidate, supervised-only**.

The project can be shown as a desktop/operator cockpit with backend-connected
read models, proposal queue, supervised fader/gain apply UI, macOS package
artifacts, and one narrow real WING fader proof. It must not be described as
autonomous/live-ready.

## What Can Be Shown

- Dark high-tech Automixer cockpit with 16 operator routes.
- Product Layer read-only snapshots for dashboard, channels, agents, groups,
  memory, training, rules, communications, topology, processing, and queue.
- Mode model: Analyze, Assist, Supervised, Auto locked/simulated.
- Supervised proposal flow that routes apply through `apply_operator_proposal`.
- Emergency stop/reset-required state in backend and frontend.
- Hardware proof/evidence panel showing readiness and blocked state.
- macOS/Electron package containing frontend build, Electron entrypoint,
  backend executable, and config resources.

## Strict Boundaries

- No direct frontend OSC writes.
- No frontend `set_fader`, `set_gain`, EQ/compressor, routing, FX, phase,
  snapshot, training, or rule writes as part of cockpit workflow.
- No direct voice-command writes under the WING deployment boundary; voice
  write intents must become Product Layer proposals first.
- No autonomous WING apply.
- EQ, compressor, phase, delay, routing, snapshots, scenes, training writes,
  and rule writes remain read-only, fail-closed, or proposal-only until a
  separate safety contract exists.
- `.env`, tokens, credentials, and local secrets are not package payload.

## Current Hardware Claim

Allowed wording:

**Supervised hardware-proven for one fader proposal path.**

Evidence:

- Proposal created through Product Layer.
- Apply sent only through `apply_operator_proposal`.
- Readback verified from cache after apply.
- Rollback returned the fader to the original value.
- Rollback readback verified from cache.
- Emergency stop blocked a later apply.

Not proven:

- gain proof;
- physical non-cache readback helper;
- autonomous operation;
- EQ/routing/snapshot live write safety;
- sustained show-length reliability.

## Dirty Worktree Handoff

Do not commit the whole tree as one slice. Stage/review these groups separately:

1. Frontend cockpit/package/DTO slice.
2. Product Layer/backend snapshot and proposal slice.
3. WING validation runner and tests.
4. Safety gate/emergency reset slice.
5. Docs/reports/screenshots.

Current staging map: `Docs/reports/AUTOMIXER_STAGING_GROUPS_20260521.md`.

Before a release candidate commit or PR, run:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
npm --prefix frontend run test:e2e
npm --prefix frontend run build
npm --prefix frontend run dist
git diff --check
```

Current verification after this handoff slice:

- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`:
  `1075 passed, 1 skipped`.
- `npx tsc --noEmit` in `frontend/`: passed.
- `npm --prefix frontend run test:e2e`: `8 passed`.
- `npm --prefix frontend test -- --watch=false`: `12 passed`.
- `npm --prefix frontend run build`: passed.
- `npm --prefix frontend run dist`: passed.
- Packaged arm64 app contains executable backend server and config resources.
- `git diff --check`: passed.

Additional Phase 1 safety-slice verification on 2026-05-21:

- Focused backend safety/product tests:
  `PYTHONPATH=backend python -m pytest tests/test_server.py::TestWingDeploymentWriteBoundary tests/test_gain_proof_dry_run_runner.py tests/test_operator_product_state.py -q`
  passed: `99 passed`.
- Full backend suite was attempted in the current sandbox and stopped after
  `271 passed` at `tests/test_gpt_control_bridge.py::test_no_write_routes_exist`
  because local HTTP bind to `127.0.0.1:0` is denied with
  `PermissionError: [Errno 1] Operation not permitted`.
- `npx tsc --noEmit` in `frontend/`: passed.
- `npm --prefix frontend test -- --watch=false`: `13 passed`.
- `npm --prefix frontend run build`: passed.
- `npm --prefix frontend run test:e2e` was attempted and blocked by the same
  sandbox class: Vite could not listen on `127.0.0.1:3000`
  (`Error: listen EPERM`).
- `git diff --check`: passed.

Next-actions execution on 2026-05-21:

- See `Docs/reports/AUTOMIXER_NEXT_ACTIONS_EXECUTION_20260521.md`.
- Full backend suite now passes in the unrestricted local environment:
  `1126 passed, 1 skipped`.
- Frontend e2e now passes in the unrestricted local environment: `8 passed`.
- `npm --prefix frontend run dist:frontend` now uses app directory plus zip for
  package smoke, avoiding the earlier DMG-only `hdiutil` cleanup failure.
- Arm64 packaged app resources were rechecked: backend executable, frontend
  `app.asar`, and required config files are present.
- Package resource scan found no `.env`, `*secret*`, `*token*`, or
  `*credential*` file names.
- Arm64 `.app` is Apple Development signed with hardened runtime; `spctl`
  rejects it for distribution because it is not a notarized Developer ID
  release artifact.
- The fake gain proof runner passed. See
  `Docs/reports/AUTOMIXER_GAIN_PROOF_PREFLIGHT_20260521.md`.

## Next Required Step

Keep the live system in supervised-only mode and run the next hardware proof
only as a separate gain trim-down validation on one channel, after explicit
safe-state reset, old-value pre-read, apply readback, rollback readback, blocked
proof, cooldown, and artifact preservation.
