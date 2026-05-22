# Automixer Staging Groups - 2026-05-21

## Purpose

The worktree is intentionally dirty across multiple slices. Do not stage or
commit the entire tree as one change. Review and stage by these groups.

## Recommended Groups

1. **Frontend cockpit and visual prototype**
   - `frontend/src/AutomixerPrototype.tsx`
   - `frontend/src/components/prototype/`
   - `frontend/src/views/`
   - `frontend/src/mock/`
   - `frontend/e2e/`
   - frontend styling/config files

2. **Frontend backend adapter and websocket layer**
   - `frontend/src/services/useOperatorBackend.ts`
   - `frontend/src/services/operatorAdapters.ts`
   - `frontend/src/services/websocket*`
   - related frontend service tests

3. **macOS/Electron package**
   - `frontend/package.json`
   - `frontend/package-lock.json`
   - `frontend/public/electron.js`
   - `build.sh`
   - `build_backend.sh`
   - `backend/server.spec`

4. **Backend Product Layer and proposal read models**
   - Product state handlers/builders
   - operator proposal queue/state/analysis changes
   - processing registry/preview/adapters
   - Product Layer DTO docs

5. **Safety gate and write-surface quarantine**
   - manual write supervision/emergency reset
   - routing/FX/snapshot/voice-command fail-closed surfaces
   - focused backend safety tests

6. **WING validation runners**
   - dry-run gain proof runner
   - supervised smoke/validation runners
   - runner tests and artifacts metadata

7. **Docs/reports/screenshots**
   - implementation plan and index updates
   - UI reference inventory and screenshots
   - handoff, readiness, Phase 1, and live-proof reports

## Pre-Commit Checks

Run before staging a release-candidate slice:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
npm --prefix frontend run test:e2e
npm --prefix frontend test -- --watch=false
npm --prefix frontend run build
npm --prefix frontend run dist:frontend
git diff --check
```

## Live Boundary

No staging group grants new live WING write approval. The only approved MVP
path remains `apply_operator_proposal` through the supervised manual gate for
`fader`/`gain`.
