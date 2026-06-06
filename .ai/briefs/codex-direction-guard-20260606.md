# Brief: Codex Direction Guard

## Task

Create repo-local automation that checks whether Codex work is moving Automixer
toward the current supervised WING Rack MVP instead of drifting into broad
memory, queue, result-consumer, report-only, or proposal-only layers.

## Constraints

- Read-only automation only.
- No WING/OSC/WebSocket live action.
- No background daemon installation.
- Must preserve existing dirty worktree changes from other Codex cycles.
- Must activate inside each Codex cycle before task selection and before the
  final report.

## Success

- A guard command produces a machine-readable and human-readable direction
  report.
- The autoloop prompt requires Codex to run and obey the guard.
- Tests cover on-target, drifting, and mixed recent work.
