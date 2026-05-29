# Agent Core Rules

## Purpose

Agent Core is the shared operating layer for Codex.app, Codex CLI, Kimi CLI,
Paperclip-facing automation, and human contributors. It defines how an agent
works before any Automixer-specific detail is used.

## Required Startup Reads

- `AGENTS.md`
- `CLAUDE.md`
- `.ai/README.md`
- `.ai/memory/project.md`
- `.ai/agent-core/safety.md`
- `.ai/project-layer/automixer_goal.md`
- `.ai/project-layer/live_safety_rules.md`

When a task touches architecture, tools, OSC, WING, Dante, gain, trim, faders,
or Paperclip, also read the matching files in `Docs/` and `.ai/project-layer/`.

## Operating Rules

- Work from files, tests, reports, and current git state.
- Preserve unrelated local changes.
- Keep diffs small and reversible.
- Do not edit production mixer logic during planning or review phases.
- Do not add dependencies unless an ADR or proposal records the reason.
- Prefer documentation, tests, audit-only tools, or dry-run utilities when the
  task is about process, memory, or safety.

## Write Boundaries

- Never touch secrets, tokens, `.env`, credentials, or private keys.
- Never perform live OSC writes, WING runtime writes, Dante routing changes, or
  gain/trim/fader mutations without explicit operator approval for the exact
  action.
- Treat recommendations, proposals, queue entries, and live control commands as
  different artifacts.
- A supervised write must have pre-read, safety check, bounded delta, operator
  approval, readback, and rollback.

## Report Duties

- Record important decisions in `.ai/project-layer/decisions.md` or `Docs/adr/`
  depending on scope.
- Record experiments in `.ai/project-layer/experiments.md`.
- Record completed work cycles in `.ai/project-layer/session_reports.md` and, if
  Paperclip needs a human-facing artifact, `.paperclip/reports/`.
- Run `python3 tools/memory/memory_lint.py` after changing memory files.
