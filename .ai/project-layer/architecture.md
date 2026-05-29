# Automixer Memory Architecture

## Purpose

This file maps the long-term memory structure to the existing Automixer
architecture and Council workflow.

## Layers

Agent Core is the shared rules layer. It defines how agents behave in any task:
read first, preserve safety, protect unrelated changes, test, report, and lint.

Project Layer is the Automixer knowledge layer. It records the live-sound goal,
WING/OSC policy, gain-staging method, safety rules, current tasks, decisions,
experiments, reports, and glossary.

Pipeline is the memory workflow layer. It defines CAPTURE, CLASSIFY, PROPOSE,
APPROVE, APPLY, LOG, and LINT, plus Codex and Paperclip cycles.

Reports are human-facing summaries in `.paperclip/reports/` and `Docs/reports/`.
They explain what happened, what passed, what failed, and what remains risky.

Tests prove that the memory pipeline exists, scripts append instead of
overwriting, lint runs, and safety rules remain explicit.

## Repository Fit

The memory layer extends existing files:

- `.ai/README.md`
- `.ai/memory/project.md`
- `.ai/briefs/`
- `.ai/proposals/`
- `.ai/reviews/`
- `.ai/decisions/`
- `Docs/adr/`
- `.paperclip/reports/`

It does not replace the Council protocol. It gives agents a stable read path
before Council proposals, implementation, review, or Paperclip reporting.

## Runtime Boundary

Memory files and tools are documentation or append-only helpers. They do not
connect to WING, OSC, Dante, audio capture, WebSocket handlers, or live runtime
state.
