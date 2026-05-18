# AUTO-MIXER Phase 1 Generated Cleanup Preview

Date: 2026-05-18
Status: preview only

No files were deleted during this phase. No WING/OSC writes, Paperclip
dispatches, issue mutations, or live runtime activation were performed.

## Verdict

Do not run a broad cleanup command in this checkout.

Both generic previews are unsafe:

- `git clean -nd` would remove real untracked source/docs from the current
  Product Layer and structure work.
- `git clean -ndX` would remove large generated folders, but also reported
  broad untracked directories because they contain ignored cache files.

Cleanup must be path-scoped.

## Current Branch State

- Branch: `main...origin/main [ahead 62, behind 26]`
- Working tree: dirty
- Current untracked source/docs include Product Layer files, tests, and
  structure documents.

## Safe Cleanup Candidates

These are generated/local dependency areas. They can be removed later with
explicit approval and path-scoped commands.

| Path | Approx size | Status |
| --- | ---: | --- |
| `backend/venv/` | 1.9G | local Python environment |
| `frontend/node_modules/` | 952M | local Node dependencies |
| `frontend/dist/` | 647M | Electron/package output |
| `backend/build/` | 394M | PyInstaller/build output |
| `backend/dist/` | 287M | packaged backend output |
| `.chromadb/` | 25M | local vector index |
| `tools/mcp_director/` | 24M | local MCP/node dependency surface |
| `tools/mcp_remote_gateway/` | 24M | local MCP/node dependency surface |
| `frontend/build/` | 1.1M | React production build output |
| `.pytest_cache/` | 180K | pytest cache |
| `__pycache__/` and nested `__pycache__/` | many files | Python bytecode cache |
| `.DS_Store` files | small | macOS metadata |

Important: `.paperclip/reports/` and `artifacts/wing_*` contain evidence and
status reports. Do not delete them as part of generic cleanup.

## Unsafe Cleanup Candidates

Do not remove these from a generic cleanup preview:

| Path | Reason |
| --- | --- |
| `Docs/PROJECT_IMPLEMENTATION_PLAN.md` | canonical implementation plan |
| `Docs/PROJECT_STRUCTURE_MAP.md` | canonical structure map |
| `Docs/PROJECT_CLEANUP_BACKLOG.md` | cleanup plan |
| `Docs/project_structure_index.json` | machine-readable structure index |
| `backend/operator_*.py` | current Product Layer source |
| `backend/handlers/product_state_handlers.py` | Product Layer websocket handlers |
| `tests/test_operator_product_state.py` | Product Layer tests |
| `automixer/logs/*.py` | source files that were accidentally ignored by broad `logs/` rule |

## Hygiene Fix Applied

`.gitignore` previously had:

```gitignore
logs/
```

In Git ignore syntax this can match nested `logs` directories too, so it ignored
`automixer/logs/__init__.py` and `automixer/logs/human_logger.py`.

The rule is now scoped:

```gitignore
/logs/
/backend/logs/
```

This keeps root/backend runtime logs ignored while allowing `automixer/logs/`
to be tracked as source.

## Recommended Cleanup Commands

Only after explicit approval:

```bash
rm -rf backend/venv frontend/node_modules frontend/dist frontend/build backend/build backend/dist
find . -type d -name __pycache__ -prune -print
find . -name .DS_Store -print
```

For the first real cleanup batch, prefer removing dependency/build outputs only:

```bash
rm -rf backend/venv frontend/node_modules frontend/dist frontend/build backend/build backend/dist
```

Then run:

```bash
PYTHONPATH=backend python -m pytest tests/test_operator_product_state.py -q
npm --prefix frontend install
npm --prefix frontend run build
```

## Next Step

Review whether `automixer/logs/__init__.py` and
`automixer/logs/human_logger.py` should be added to git with the rest of the
structure/Product Layer slice. They are source files and should not be deleted
by generated cleanup.
