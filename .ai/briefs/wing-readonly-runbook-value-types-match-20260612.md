# Brief: WING read-only runbook value-types match

## Context

The WING read-only runner and no-WING release gate use
`--value-types fader,gain,pan`. The read-only runbook examples still showed
`fader,gain`, which can create a strict expected-plan mismatch when operators
copy the command.

## Task

Align the runbook command examples with the runner/no-WING template.

## Acceptance

- Plan-only and real-run examples use `--value-types fader,gain,pan`.
- Rails tests prevent reverting to `fader,gain`.
- The change does not widen write authority or contact WING.
