# Experiments

## Purpose

This file stores experiment notes that may later become decisions, tests, or
ADR entries.

## Capture Rules

- Capture raw observations first.
- Mark whether evidence is offline, audit-only, dry-run, supervised-write, or
  live-write.
- Do not claim live readiness from offline or dry-run evidence.
- Link relevant files, commands, traces, and reports.
- Promote stable lessons into `.ai/project-layer/decisions.md`,
  `.ai/memory/project.md`, or `Docs/adr/` when appropriate.

## Records

### 2026-05-29 - Memory pipeline setup

Mode: audit-only

Observation:
Automixer has enough existing Council and Paperclip structure to support a
Markdown memory pipeline without changing production mixer code.

Next:
Use memory lint and append-only helpers to keep records consistent.
