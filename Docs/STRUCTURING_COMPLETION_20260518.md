# AUTO-MIXER Structuring Completion Report

Date: 2026-05-18
Scope: project structure, documentation layout, backend root de-noising, and
agent-facing indexes. No live WING/OSC actions were performed.

## Completed

- Added `Docs/INDEX.md` as the main documentation entrypoint.
- Moved old root notes into `Docs/archive/root-notes/`.
- Moved backend documentation into `Docs/backend/`.
- Moved historical TUB reports into `Docs/reports/tub/`.
- Moved WING protocol/manual PDFs into `Docs/manuals/wing/`.
- Moved the tracked legacy generated test result into
  `Docs/archive/generated/backend_test_results_legacy.json`.
- Moved `backend/server.py.bak` into
  `backend/lab_only/legacy_backups/server.py.bak`.
- Added backend manual-probe inventory in `Docs/backend/MANUAL_PROBES.md`.
- Updated the ChromaDB project indexer to treat `Docs/reports/tub/TUB-*.md` as
  `tub_report` sources.

## Intentionally Not Moved

Manual backend probe scripts such as `backend/test_wing_connection.py`,
`backend/route_*.py`, and `backend/query_*.py` remain in `backend/` for now.
They are documented as manual probes because moving them physically would risk
breaking working-directory imports and old operator commands.

The next safe cleanup step is to migrate those probes one-by-one into
`backend/lab_only/` with compatibility wrappers or converted tests.

## Current Structure Rule

- `backend/` is primarily source code.
- `Docs/` is project knowledge.
- `Docs/reports/tub/` is historical issue/report knowledge.
- `Docs/archive/` is old or generated context.
- `backend/lab_only/` is quarantine and manual tooling.
- `.paperclip/reports/` remains local/generated.

## Remaining Risks

- Frontend dependency audit is still red and needs a separate modernization
  task.
- Local branch still diverges from `origin/main`; do not push casually.
- Physical WING readiness was not changed by this cleanup.

## Verification Target

This structuring step should be accepted only if:

- backend tests still pass;
- frontend build still passes;
- project vector index tests still pass;
- git diff has no whitespace errors;
- ignored caches and generated local logs are cleaned before final status.

## Verified

Executed after the structure changes:

| Check | Result |
| --- | --- |
| `PYTHONPATH=backend python -m pytest tests/test_project_vector_index.py -q` | `13 passed` |
| `PYTHONPATH=backend python -m pytest tests/ -q` | `743 passed, 6 skipped, 2 warnings` |
| `npm --prefix frontend run build` | Passed |

Frontend build still emits the existing Node `DEP0176 fs.F_OK` deprecation
warning from the React toolchain.
