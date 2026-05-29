# Memory Pipeline

## Purpose

The memory pipeline keeps Automixer knowledge durable without turning notes into
unsafe runtime authority.

## Layers

- Agent Core: shared rules and safety.
- Project Layer: Automixer-specific knowledge.
- Pipeline: process for capturing, classifying, approving, applying, logging,
  and linting memory.
- Reports: human-readable work summaries.
- Tests: proof that the pipeline exists and stays safe.

## Flow

CAPTURE records raw notes from work cycles, Paperclip reports, tests, telemetry,
or soundcheck observations.

CLASSIFY labels the note as task, decision, experiment, safety rule, report, or
glossary update.

PROPOSE turns important observations into a concrete memory change.

APPROVE uses human review, Council review, or ADR review for changes that affect
architecture, safety, or runtime behavior.

APPLY appends the approved memory record or creates the required Markdown file.

LOG records what changed and what evidence was used.

LINT runs `python3 tools/memory/memory_lint.py` to check required files,
sections, and safety terms.

## Safety Boundary

The memory pipeline can write Markdown and local reports. It must not write to
WING, OSC, Dante, mixer runtime state, audio capture state, or live parameters.
