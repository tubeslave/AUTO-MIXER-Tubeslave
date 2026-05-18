# AUTO-MIXER Cleanup Backlog

Date: 2026-05-18
Status: safe cleanup backlog

This is the cleanup plan for making the project clear without breaking useful
work. It is intentionally staged. Do not start by deleting code.

## Cleanup Principles

- Inventory before deletion.
- Preserve the canonical MVP path.
- Keep offline and live code separated.
- Mark experiments before archiving them.
- Keep generated files out of architecture decisions.
- Make each cleanup batch small, testable, and reversible.

## Phase 0: Baseline And Labels

Goal: make the current state understandable.

Tasks:

- Keep `Docs/PROJECT_IMPLEMENTATION_PLAN.md` as the main implementation plan.
- Use `Docs/PROJECT_STRUCTURE_MAP.md` as the domain map.
- Use `Docs/project_structure_index.json` as the machine-readable agent index.
- Record branch/diff before any cleanup batch.
- Label each module as `canonical`, `optional`, `lab`, `legacy`,
  `quarantine`, or `generated`.

Done when:

- every top-level project domain has an owner/status;
- the next cleanup batch can be reviewed without reading the whole repo.

## Phase 1: Generated Files And Local Outputs

Goal: separate source from local generated material.

Candidate paths:

- `__pycache__/`
- `.pytest_cache/`
- `frontend/build/`
- `frontend/dist/`
- `backend/build/`
- `backend/dist/`
- `backend/venv/`
- local `logs/`
- local `runtime/`
- selected local render outputs

Safe workflow:

```bash
git status --short
git clean -ndX
```

Do not run destructive cleanup until the preview is reviewed.

Done when:

- ignored/generated files are clearly separated;
- evidence artifacts under `.paperclip/reports/` and `artifacts/wing_*` are
  preserved unless explicitly archived.

## Phase 2: Backend Root De-noising

Goal: reduce confusion in `backend/` without changing behavior.

Current issue:

`backend/` contains production modules, old manual scripts, test-like scripts,
TUB reports, voice experiments, routing utilities, and live write utilities in
one folder.

Candidate groups:

| Group | Examples | Target status |
| --- | --- | --- |
| Runtime source | `server.py`, `wing_client.py`, `mixer_client_base.py` | `canonical` |
| Product Layer | `operator_*.py`, `handlers/product_state_handlers.py` | `canonical` |
| Automation modules | `auto_eq.py`, `auto_fader.py`, `auto_soundcheck_engine.py` | `canonical` or `optional` |
| Manual WING scripts | `route_*.py`, `query_*.py`, `list_snapshots.py` | `quarantine` until reviewed |
| Old test scripts in backend root | `test_*.py` | `legacy` until moved or replaced |
| Voice variants | `voice_control*.py` | `lab` / `optional` |
| TUB reports | `TUB-*.md` | docs/report archive candidate |

Safe workflow:

- First add status labels in documentation.
- Then move only clearly non-runtime scripts into a reviewed namespace.
- Keep import-compatible shims if anything imports the old path.
- Run focused tests after each small move.

Done when:

- a developer can open `backend/` and identify runtime source quickly;
- manual live scripts cannot be mistaken for canonical runtime.

## Phase 3: Live Write Path Review

Goal: make live-console safety mechanically obvious.

Candidate surfaces:

- `backend/wing_client.py`
- `backend/server.py`
- `backend/auto_soundcheck_engine.py`
- `backend/auto_eq.py`
- `backend/auto_fader.py`
- `backend/phase_alignment.py`
- `backend/handlers/snapshot_handlers.py`
- manual route/snapshot/reset scripts

Required classification for each write-capable path:

- read-only;
- proposal-only;
- supervised live write;
- blocked/quarantined;
- lab/manual operator script.

Done when:

- unapproved write attempts fail closed;
- blocked write result is propagated to callers;
- success logs cannot hide blocked writes;
- rollback/readback evidence exists for approved supervised writes.

## Phase 4: Offline Pipeline Consolidation

Goal: keep one accepted offline path clear while preserving useful experiments.

Canonical/offline:

- `automixer/production_mix_v1/`
- `config/pipelines/production_mix_v1.yaml` if present

Optional/legacy:

- `mix_agent/`
- `ai_mixing_pipeline/`
- older render utilities and candidate-output folders

Done when:

- the accepted offline entrypoint is documented;
- old offline agents are either linked as optional support or marked legacy;
- no offline path imports live WING clients unless explicitly mocked/tested.

## Phase 5: Research And ML Boundary

Goal: keep research useful but non-invasive.

Candidate paths:

- `external/`
- `backend/ml/`
- `models/training_datasets/`
- `backend/perceptual/`
- `backend/evaluation/`

Done when:

- research modules are not imported by live runtime by accident;
- heavy assets are excluded from packaging unless intentionally needed;
- evaluation utilities produce compact reports, not hidden behavior changes.

## Phase 6: Documentation Consolidation

Goal: reduce contradictory docs.

Canonical docs:

- `Docs/PROJECT_IMPLEMENTATION_PLAN.md`
- `Docs/PROJECT_STRUCTURE_MAP.md`
- `Docs/PROJECT_CLEANUP_BACKLOG.md`
- `Docs/CONVENTIONS.md`
- `Docs/ARCHITECTURE.md` after refresh

Archive candidates:

- old one-off backend `TUB-*.md` files after their facts are summarized;
- stale quick-start docs that mention unsupported default behavior;
- duplicate voice/routing test notes.

Done when:

- docs match current config and tests;
- agents know which documents are canonical;
- old docs are marked as archive material instead of silently trusted.

## Recommended First Cleanup Batch

Do this before any broad restructuring:

1. Review the current Product Layer dirty slice.
2. Run focused backend tests for operator product state.
3. Run frontend build.
4. Confirm no live write path was added.
5. Publish a short Product Layer status report.

Why first:

- this slice is already in the working tree;
- it directly supports the MVP;
- it avoids risky file moves before product state is stable.

## Commands For Review

```bash
git status --short --branch
PYTHONPATH=backend python -m pytest tests/test_operator_product_state.py -q
npm --prefix frontend run build
git grep -n '<<<<<<<\|>>>>>>>' -- ':!external/**' ':!frontend/node_modules/**' ':!backend/venv/**'
```
