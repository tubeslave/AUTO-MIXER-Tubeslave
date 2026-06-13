# Brief: WING handoff command artifact binding

## Context

The WING read-only handoff already declares expected artifact globs and compares
them with the offline plan. The operator-visible no-WING summary should also
show that the capture, artifact review, and decision review commands are bound
to the same expected plan, artifact root, and decision root before any real WING
session.

## Task

Expose and validate command-level artifact binding for the read-only handoff
packet summary.

## Acceptance

- The handoff validator reports capture command `--expected-plan` and
  `--artifact-root` bindings.
- The handoff validator reports review and decision-review command
  `--artifact-root` bindings, plus decision-review `--decision-root`.
- Mismatched command bindings fail no-WING validation.
- Project Director, WING rails, runbook, and MVP docs surface the new guard.
- No backend, WING, OSC, or live-write behavior is added.
