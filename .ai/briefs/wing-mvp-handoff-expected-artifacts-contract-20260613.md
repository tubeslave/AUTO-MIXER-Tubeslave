# WING MVP Handoff Expected Artifacts Contract

## Context

The WING read-only preflight expected plan declares the summary and event-log
artifact paths it expects from a real run. The handoff packet already carries
named summary/event globs, but the packet did not expose or validate a unified
`expected_artifacts` list against the expected plan artifact.

## Task

Make the WING read-only preflight handoff packet expose the expected artifact
list and make the no-WING release gate verify that the handoff list and named
summary/event globs match the expected plan.

## Constraints

- No backend startup.
- No WING transport usage.
- No OSC/WebSocket writes.
- No readiness closure from offline packet evidence.
- Keep the supervised WING MVP blocker boundary intact.

## Test Plan

- Focused handoff/no-WING pytest modules.
- no-WING quick gate with a local artifact root.
- Project direction and WING rail tests.
- `git diff --check`.
- `tools/memory/memory_lint.py`.
- Standard pytest suite.
