# AUTO-MIXER Documentation Index

Date: 2026-05-18
Status: canonical documentation entrypoint

Start here before changing the project structure, runtime behavior, WING/OSC
paths, Product Layer, or cleanup policy.

## First Read

| Document | Purpose |
| --- | --- |
| `Docs/PROJECT_IMPLEMENTATION_PLAN.md` | Product goal, MVP phases, safety boundaries, schedule. |
| `Docs/PROJECT_STRUCTURE_MAP.md` | Canonical domain map and status labels. |
| `Docs/HEALTH_BASELINE_20260518.md` | Current post-cleanup test/build/audit baseline. |
| `Docs/STRUCTURING_COMPLETION_20260518.md` | What was moved, what remains, and the next work boundary. |
| `Docs/PROJECT_CLEANUP_BACKLOG.md` | Remaining cleanup backlog and rules. |

## Runtime And Safety

| Area | Documents |
| --- | --- |
| Runtime hygiene | `Docs/RUNTIME_HYGIENE.md` |
| Live readiness and safety reports | `Docs/reports/tub/` |
| WING manuals and protocol PDFs | `Docs/manuals/wing/` |
| Secret rotation | `Docs/secrets-rotation.md` |

## Backend

| Area | Documents |
| --- | --- |
| Backend docs | `Docs/backend/` |
| Backend entrypoint guide | `Docs/backend/README.md` |
| Manual probes and legacy checks | `Docs/backend/MANUAL_PROBES.md` |
| Backend source package | `backend/` |
| Legacy backup quarantine | `backend/lab_only/legacy_backups/` |

## Product And Agents

| Area | Documents |
| --- | --- |
| Product Layer plan | `Docs/PROJECT_IMPLEMENTATION_PLAN.md` |
| Paperclip/GPT workflow | `Docs/PAPERCLIP_DIRECTOR_WORKFLOW.md` |
| Operational memory | `Docs/OPERATIONAL_MEMORY.md` |
| Tooling overview | `Docs/TOOLS.md` |
| Local Paperclip-readable reports | `.paperclip/reports/` |

## Archive

| Area | Path |
| --- | --- |
| Old root notes | `Docs/archive/root-notes/` |
| Generated legacy snapshots | `Docs/archive/generated/` |
| Old TUB reports | `Docs/reports/tub/` |

## Rules

- `backend/` is for source code and runtime-adjacent scripts, not report
  storage.
- `Docs/` is the source of truth for project knowledge.
- `Docs/reports/tub/` keeps historical issue reports searchable without
  cluttering runtime folders.
- `backend/lab_only/` is for quarantine, legacy, and manual tooling that should
  not be imported by live runtime.
- No live WING/OSC action is approved by documentation changes.
