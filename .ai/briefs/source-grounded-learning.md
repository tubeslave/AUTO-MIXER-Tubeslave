# Source-Grounded Learning

## Task

Add the first infrastructure layer for learning mixing decisions from authoritative
sources: books, papers, standards, and reviewed video/channel sources.

## Goals

- Keep live OSC behavior unchanged by default.
- Seed an auditable source registry and atomic rule dataset.
- Add retrieval by problem/domain/instrument.
- Add JSONL logging for decisions and human feedback.
- Make the layer usable later by MERT/Codex evaluators without making them the
  authority source.

## Non-Goals

- No automatic influence on live mixer actions in this first step.
- No new production dependencies.
- No copyrighted long excerpts from books or videos.
- No MERT/Codex policy integration yet.

## Test Plan

- Source registry and rules load.
- Every seeded rule references known sources.
- Search returns relevant rules.
- JSONL logging writes decision and feedback events.
- Config defaults keep the layer disabled.
