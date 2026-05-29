# Agent Core Source Policy

## Purpose

Source policy defines what agents trust when they reason about Automixer.

## Source Priority

1. Current checkout files and current git diff.
2. `AGENTS.md`, `CLAUDE.md`, and `.ai/agent-core/`.
3. `.ai/project-layer/` current project memory.
4. `Docs/`, especially architecture, conventions, tools, reports, and ADRs.
5. Tests and recent test output.
6. `.paperclip/reports/` for human-readable cycle reports.
7. External knowledge only when explicitly requested or needed for current
   standards, product docs, or version-sensitive facts.

## Hardware Sources

For WING/OSC details, prefer:

- `Docs/TOOLS.md`
- `Docs/Behringer_WING_Official/`
- `Docs/manuals/wing/`
- current `backend/wing_client.py` behavior
- telemetry artifacts that clearly state whether they are live, dry-run, or
  audit-only

## Stale Evidence

Treat old reports as context, not proof of current behavior. Re-run tests or
inspect current code when the result affects safety, architecture, or runtime
claims.

## Report Source Rules

Paperclip reports should cite paths, commands, and observed outcomes. They
should not claim live readiness unless dry-run, pre-read, readback, and rollback
evidence exist for the exact path.
