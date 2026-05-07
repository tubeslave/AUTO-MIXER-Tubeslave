# Task audio-training-datasets: connect trainer agent to dataset discovery

## Problem

The standalone `trainer_agent` exists, but it only wraps a generic training command. The current task is to analyze the research report on audio-mixing datasets, connect the created sound-engineering training agent to a concrete dataset-discovery/download step, and start a safe first download pass.

## Constraints

- Do not touch mixer control paths, live faders, OSC/MIDI commands, or secrets.
- Avoid new production dependencies.
- Keep initial downloads small and legally cautious.
- Prefer metadata/dialogue/parameter datasets before large audio archives.
- Record the decision in `Docs/adr/`.

## Definition of Done

- `trainer_agent` can run dataset discovery through a first-class task.
- The runner can trigger that task with the downloaded research report.
- Dataset discovery writes a manifest and downloads compact public Hugging Face files.
- Tests cover command construction and safe file-selection behavior.

## Test Command

`PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
