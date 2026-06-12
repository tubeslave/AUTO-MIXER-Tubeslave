# ADR: WING read-only handoff value types summary

## Context

The WING read-only runner plan and capture commands explicitly use
`--value-types fader,gain,pan`. The handoff command used the same value types
through the handoff tool default, but the no-WING release summary did not show
or validate that value-type alignment.

## Options Considered

1. Continue relying on handoff defaults.
2. Pass value types explicitly and validate the handoff/capture match in
   no-WING evidence.

## Decision

Pass `--value-types fader,gain,pan` through the handoff command and report the
match in no-WING evidence.

## Why This Won

Expected-plan equality is intentionally strict. The operator-facing handoff
path should make the value-type set visible before hardware-session prep.

## Rejected Alternatives

Relying only on defaults was rejected because it hides a field that participates
in the read-only request plan.

## Implementation Plan

- Update the handoff command builder in `tools/no_wing_release_check.py`.
- Surface handoff value types in no-WING JSON/Markdown summary.
- Update runbook and tests.

## Test Plan

- Focused no-WING release-check tests.
- WING MVP rails tests.
- Full repository pytest.

## Risks And Rollback

The change does not contact WING, backend WebSocket, or OSC. Rollback is to
remove the explicit handoff value-types flag and summary field.
