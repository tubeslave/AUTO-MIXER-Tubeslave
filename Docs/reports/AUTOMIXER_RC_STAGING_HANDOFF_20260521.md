# Automixer RC Staging Handoff - 2026-05-21

## Verdict

Status: **release-candidate staging ready, supervised-only, no new live writes**.

This handoff implements the next necessary non-live action from the global
plan: make the current dirty worktree reviewable, preserve the live boundary,
and define the exact next WING validation step. It does not run physical WING
gain proof and does not broaden the approved write path.

Allowed current live claim remains:

**Supervised hardware-proven for one fader proposal path.**

## Current Evidence

Local verification already recorded in
`Docs/reports/AUTOMIXER_NEXT_ACTIONS_EXECUTION_20260521.md`:

- Full backend suite: `1126 passed, 1 skipped`.
- Frontend e2e: `8 passed`.
- Frontend unit tests: `13 passed`.
- Frontend typecheck and build: passed.
- `npm --prefix frontend run dist:frontend`: passed with app directory plus zip.
- `git diff --check`: passed.
- Fake gain proof dry-run: passed.

Package/resource checks from this handoff pass:

- Packaged backend executable exists and is executable:
  `frontend/dist/mac-arm64/AUTO MIXER Tubeslave.app/Contents/Resources/backend/server`.
- Required config resources exist in the packaged app:
  `default_config.json`, `automixer.yaml`, `muq_eval.yaml`,
  `perceptual.yaml`, `source_knowledge.yaml`.
- `app.asar` contains `/package.json`, `/public/electron.js`, and
  `/build/index.html`.
- Package resource scan found no `.env`, `*secret*`, `*token*`, or
  `*credential*` file names.

Verification performed for this handoff execution:

- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`:
  `1126 passed, 1 skipped`.
- `npx tsc --noEmit` in `frontend/`: passed.
- `npm --prefix frontend test -- --watch=false`: `13 passed`.
- `npm --prefix frontend run test:e2e`: `8 passed`.
- `npm --prefix frontend run build`: passed.
- `npm --prefix frontend run dist:frontend`: passed.
- `git diff --check`: passed.

Local secret note:

- A local root `.env` file exists and is ignored by `.gitignore`.
- It must not be staged, copied into package resources, or used as release
  evidence.

macOS signing/package note:

- The arm64 app is Apple Development signed with hardened runtime.
- `spctl --assess` rejects it for distribution. This is expected until a
  Developer ID signed and notarized release artifact is produced.
- Current app/zip artifacts are valid for local smoke/handoff, not public
  distribution.

## Staging Decision

Do not stage or commit the entire worktree as one change. Use this order:

1. **Docs and handoff slice**
   - Reports, index updates, implementation plan wording, screenshot indexes.
   - Purpose: make current status visible to Codex/Kimi/Paperclip before code
     review.

2. **macOS/Electron package slice**
   - Package scripts, Electron entrypoint, PyInstaller spec, build scripts.
   - Purpose: preserve the verified app bundle shape.

3. **Backend Product Layer and safety slice**
   - Product Layer snapshots, proposal queue/state, operator mode, emergency
     stop/reset-required state, tests.
   - Purpose: keep all write-capable paths proposal-gated or fail-closed.

4. **Frontend cockpit and adapter slice**
   - Cockpit routes, backend adapter, websocket methods, fixtures, e2e.
   - Purpose: keep UI work separate from live-write policy.

5. **WING validation runner slice**
   - Dry-run and supervised validation runners, runner tests, artifact metadata.
   - Purpose: preserve proof tooling without mixing it into application runtime.

If a file appears in more than one conceptual slice, review it manually before
staging. Do not use bulk `git add .` for release-candidate work.

## Live Boundary

No new physical WING write is approved by this handoff.

Allowed MVP write path:

```text
Product Layer proposal -> apply_operator_proposal -> supervised manual gate -> readback -> rollback -> evidence
```

Forbidden until a separate safety contract exists:

- direct `set_gain` or `set_fader` messages;
- EQ, compressor, routing, FX, phase/delay, snapshot, scene, training, or rule
  writes;
- Auto mode live apply;
- autonomous agent apply;
- multi-channel hardware proof.

## Next Required Live Step

The next hardware step is a separate physical `gain` proof:

1. WING connected and read-only preflight green.
2. Mode is `Supervised`.
3. Emergency stop/reset-required state is explicitly cleared by the operator.
4. One low-risk channel is selected.
5. Current gain is physically/readably captured.
6. One small gain trim-down proposal is selected, default `-0.25 dB`.
7. Apply uses only `apply_operator_proposal`.
8. Apply readback is verified.
9. Rollback is executed through the supervised path.
10. Rollback readback is verified.
11. Emergency-stop blocked proof is captured.
12. Artifacts/logs are preserved.

If any readback or rollback proof is missing, status remains
`partial/blocked`.

Success wording after a fully passing live gain proof:

**Supervised hardware-proven for fader and one gain trim-down path.**

## Pre-Commit Checks

Run before staging each release-candidate slice:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
npm --prefix frontend run test:e2e
npm --prefix frontend test -- --watch=false
npm --prefix frontend run build
npm --prefix frontend run dist:frontend
git diff --check
```
