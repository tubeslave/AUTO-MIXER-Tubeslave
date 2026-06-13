# Brief: WING direction guard skips generated guard reports

## Context

`codex_direction_guard.py` scans recent reports as evidence. Generated direction
guard reports are derivative outputs and can contain prior evidence summaries.
They should not become fresh evidence in later cycles.

## Task

Exclude generated direction guard reports from evidence scoring.

## Acceptance

- Generated guard Markdown and JSON reports do not create target, drift, or
  mixed evidence.
- Real WING read-only proof evidence still scores normally.
- Project Director remains on the selected WING read-only blocker.
- No backend, WING, OSC, or live-write behavior is added.
