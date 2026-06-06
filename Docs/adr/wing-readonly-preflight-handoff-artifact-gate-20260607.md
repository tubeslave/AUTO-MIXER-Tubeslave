# ADR: WING Read-Only Preflight Handoff Artifact Gate

## Context

The supervised WING proof path depends on operators running a read-only
preflight, reviewing the resulting summary/events artifacts, and recording a
decision before any supervised fader/gain proof. The repository already builds
an operator handoff packet, but the no-WING release check did not validate the
generated packet as structured evidence.

## Options considered

1. Keep handoff generation as Markdown-only smoke output.
2. Add a separate validator CLI for handoff packets.
3. Generate JSON in the no-WING run and validate it inside
   `tools/no_wing_release_check.py`.

## Decision

Use option 3. The handoff tool keeps Markdown as its default for operator use,
but no-WING now writes a JSON packet in the run directory and validates it as
`wing_readonly_preflight_handoff_packet_artifact_schema`.

## Why this won

The handoff packet is part of the proof boundary. Validating the exact generated
run artifact catches drift in commands, channel scope, plan output, physical
readback requirements, review-template output, and safety flags without adding
another live or hardware-dependent component.

## Rejected alternatives

- Markdown-only output is useful for humans but too weak as release evidence.
- A standalone validator CLI would duplicate no-WING release policy.
- Validating only the static `Docs/reports` default path would not prove the
  artifact produced by the active release-check run.

## Implementation plan

1. Make no-WING call the handoff tool with `--format json`.
2. Store the artifact as `wing_readonly_preflight_handoff_packet.json`.
3. Add a schema validator for safety flags, commands, checklist IDs, blockers,
   and plan-output consistency.
4. Append the validator result after the handoff command in release checks.
5. Expose the artifact summary in `summary.json` and `summary.md`.

## Test plan

- Unit tests for valid, missing, and unsafe handoff packet artifacts.
- Existing handoff packet tests continue to cover Markdown/default behavior.
- no-WING quick run verifies the validator in the release-check sequence.
- Memory lint and diff check verify docs/tooling hygiene.

## Risks and rollback

The risk is confusing a valid handoff packet with real hardware proof. The gate
therefore keeps readiness update, readiness closure, live authority, and full
autonomous flags false. Rollback is to remove the JSON handoff output, schema
validator, tests, and report/index additions; no mixer transport path is
affected.
