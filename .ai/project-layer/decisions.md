# Decisions

## Purpose

This file records project-layer decisions that are useful for future agents.
Architecture-significant decisions still belong in `Docs/adr/`.

## Decision Index

- 2026-05-29: Add Markdown memory pipeline as documentation plus append-only
  utilities. Runtime mixer code remains unchanged.

## Records

### 2026-05-29 - Markdown memory pipeline

Status: accepted

Context:
Automixer already had AI Council memory, Paperclip reports, ADRs, and safety
rules. It needed a clearer long-term Markdown memory layer shaped as Agent Core,
Project Layer, and Pipeline.

Decision:
Add `.ai/agent-core/`, `.ai/project-layer/`, `.ai/pipeline/`, `tools/memory/`,
focused tests, and `Docs/memory_pipeline.md`.

Why:
This gives Codex, Kimi, Paperclip, and humans a stable read/write path without
changing live mixer behavior.

Safety:
The pipeline is documentation and append-only tooling. It performs no OSC, WING,
Dante, gain, trim, fader, routing, mute, snapshot, or live audio writes.

Tests:
`tests/test_memory_pipeline.py`

Rollback:
Remove the new memory files and tests if the project decides to keep only the
older Council memory layout.
