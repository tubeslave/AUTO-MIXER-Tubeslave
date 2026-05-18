# AUTO-MIXER Staging Boundaries

Date: 2026-05-18
Status: staging plan only

No files were staged or committed while producing this boundary plan. No
WING/OSC writes, Paperclip dispatches, issue mutations, or live runtime
activation were performed.

## Current Verdict

The current worktree should not be staged as one mixed commit.

Use three clean commit slices and one quarantine bucket:

1. Product Layer MVP slice.
2. Structure/cleanup documentation slice.
3. `.gitignore` hygiene slice.
4. Quarantine bucket: `automixer/logs/*.py`.

## Slice 1: Product Layer MVP

Purpose: make the operator workflow real:

```text
runtime snapshot -> operator analysis -> proposal queue -> Control Center
  -> supervised apply only when mode and approval allow it
```

Files:

- `backend/handlers/__init__.py`
- `backend/server.py`
- `backend/handlers/product_state_handlers.py`
- `backend/operator_analysis.py`
- `backend/operator_product_state.py`
- `backend/operator_proposal_queue.py`
- `backend/operator_recommendation_bridge.py`
- `frontend/src/App.js`
- `frontend/src/components/ControlCenterViews.css`
- `frontend/src/components/ControlCenterViews.js`
- `frontend/src/services/websocket.js`
- `tests/test_operator_product_state.py`

Verification already run:

- `PYTHONPATH=backend python -m pytest tests/test_operator_product_state.py -q`
  -> `16 passed`
- `PYTHONPATH=backend python -m pytest tests/test_server.py tests/test_operator_product_state.py -q`
  -> `80 passed`

Frontend note:

- `npm --prefix frontend run build` passed before dependency cleanup.
- It was not rerun after cleanup because `frontend/node_modules/` was
  intentionally removed.
- Reinstall with `npm --prefix frontend install` before the next frontend build.

Safety verdict:

- The Product Layer has one live apply path:
  `server._apply_manual_console_write(...)`.
- Incomplete proposals are now analysis-only and blocked before reaching the
  manual write gate.
- This slice does not prove real WING readiness.

Suggested staging command:

```bash
git add \
  backend/handlers/__init__.py \
  backend/server.py \
  backend/handlers/product_state_handlers.py \
  backend/operator_analysis.py \
  backend/operator_product_state.py \
  backend/operator_proposal_queue.py \
  backend/operator_recommendation_bridge.py \
  frontend/src/App.js \
  frontend/src/components/ControlCenterViews.css \
  frontend/src/components/ControlCenterViews.js \
  frontend/src/services/websocket.js \
  tests/test_operator_product_state.py
```

Suggested commit message:

```text
Add operator Product Layer proposal workflow
```

## Slice 2: Structure And Cleanup Documentation

Purpose: give agents and humans a canonical map of the project.

Files:

- `Docs/PROJECT_IMPLEMENTATION_PLAN.md`
- `Docs/PROJECT_STRUCTURE_MAP.md`
- `Docs/PROJECT_CLEANUP_BACKLOG.md`
- `Docs/project_structure_index.json`
- `Docs/PHASE1_GENERATED_CLEANUP_PREVIEW.md`

Verification already run:

- `python -m json.tool Docs/project_structure_index.json`
- `PYTHONPATH=backend python -m pytest tests/test_operator_product_state.py -q`
  -> `16 passed`

Suggested staging command:

```bash
git add \
  Docs/PROJECT_IMPLEMENTATION_PLAN.md \
  Docs/PROJECT_STRUCTURE_MAP.md \
  Docs/PROJECT_CLEANUP_BACKLOG.md \
  Docs/project_structure_index.json \
  Docs/PHASE1_GENERATED_CLEANUP_PREVIEW.md
```

Suggested commit message:

```text
Document Automixer structure and cleanup boundaries
```

## Slice 3: Gitignore Hygiene

Purpose: stop hiding source-like nested `logs` folders while preserving root and
backend runtime logs as ignored local output.

File:

- `.gitignore`

Change:

```gitignore
/logs/
/backend/logs/
```

replaces the broad:

```gitignore
logs/
```

Suggested staging command:

```bash
git add .gitignore
```

Suggested commit message:

```text
Scope log ignore rules to runtime log folders
```

## Quarantine Bucket: Do Not Stage Yet

Files:

- `automixer/logs/__init__.py`
- `automixer/logs/human_logger.py`

Reason:

- They became visible after the `.gitignore` hygiene fix.
- They are not self-contained in the current checkout.
- Import check fails:

```text
ModuleNotFoundError: No module named 'automixer.decision.models'
```

Verdict:

- Do not include them in the Product Layer commit.
- Do not include them in the structure docs commit.
- Decide separately whether to restore the missing `automixer.decision` source,
  move these files to a quarantine namespace, or delete them after review.

## Ignored Report Artifacts

The `.paperclip/reports/*.md` files produced during this work are intentionally
ignored by `.gitignore`.

They are useful local handoff artifacts, but they should not be assumed to be
part of the git commit unless the project policy changes.

## Current Local Cleanup State

Already removed with explicit approval:

- `backend/venv/`
- `frontend/node_modules/`
- `frontend/dist/`
- `frontend/build/`
- `backend/build/`
- `backend/dist/`
- `__pycache__/`
- `.pytest_cache/`
- `.DS_Store`

Running tests recreates small cache directories; this is expected.

## Recommended Next Action

Stage and commit Slice 2 first, then Slice 1, then Slice 3. Keep the quarantine
bucket out of all commits until a separate decision is made.
