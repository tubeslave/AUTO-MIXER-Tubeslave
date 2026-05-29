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
| `Docs/PROCESSING_METHODS_INVENTORY.md` | Canonical processing-method and pipeline-profile selection. |
| `Docs/HEALTH_BASELINE_20260518.md` | Current post-cleanup test/build/audit baseline. |
| `Docs/PHASE1_SUPERVISED_WRITE_PROGRESS_20260519.md` | First Phase 1 supervised-write implementation slice and verification. |
| `Docs/reports/AUTOMIXER_HANDOFF_WING_VALIDATION_20260520.md` | Current cockpit/package handoff and narrow real WING supervised proposal proof. |
| `Docs/reports/AUTOMIXER_RELEASE_CANDIDATE_HANDOFF_20260520.md` | MVP cockpit handoff boundaries, package claim, and staging groups. |
| `Docs/reports/AUTOMIXER_PHASE1_WRITE_PATH_INVENTORY_20260520.md` | Current approved, quarantined, and proposal-only live-write surfaces. |
| `Docs/reports/AUTOMIXER_SLICE3_PROOF_READINESS_HANDOFF_20260520.md` | Proof-readiness contract before any physical `gain` validation. |
| `Docs/reports/AUTOMIXER_NEXT_ACTIONS_EXECUTION_20260521.md` | Current non-live execution report for package handoff, safety checks, and next gain-proof preconditions. |
| `Docs/reports/AUTOMIXER_GAIN_PROOF_PREFLIGHT_20260521.md` | Exact preflight, allowed flow, and success wording for the next physical `gain` proof. |
| `Docs/reports/AUTOMIXER_STAGING_GROUPS_20260521.md` | Dirty-worktree staging groups for release-candidate handoff. |
| `Docs/reports/AUTOMIXER_RC_STAGING_HANDOFF_20260521.md` | Release-candidate staging order, package evidence, and strict live boundary before the physical `gain` proof. |
| `Docs/reports/AUTOMIXER_PRE_LIVE_GAIN_REVIEW_20260521.md` | Pre-live code review, focused tests, and refreshed fake gain proof before any physical `gain` validation. |
| `Docs/reports/AUTOMIXER_OFFLINE_NO_WING_EXECUTION_20260522.md` | No-WING execution slice: safety audit refresh, fake gain proof, full checks, and package smoke. |
| `Docs/reports/AUTOMIXER_AUTOMATION_FOUNDATION_LAYER_20260527.md` | Automation foundation layer: decision traces, safety monitor, replay DTOs, idempotent supervised apply, and current validation evidence. |
| `Docs/adr/automation-foundation-layer-20260527.md` | ADR for the safe agentic-mixing foundation layer and its live-write boundary. |
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
| Processing pipeline selection | `Docs/PROCESSING_METHODS_INVENTORY.md`, `Docs/adr/processing-pipeline-selection.md` |
| Automation foundation layer | `Docs/adr/automation-foundation-layer-20260527.md`, `Docs/reports/AUTOMIXER_AUTOMATION_FOUNDATION_LAYER_20260527.md` |
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
