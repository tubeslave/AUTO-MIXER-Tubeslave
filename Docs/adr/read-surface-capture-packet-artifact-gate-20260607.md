# ADR: Read-Surface Capture Packet Artifact Gate

## Context

The read-surface physical-readback capture packet is the operator handoff that
points to the next required physical artifact. The no-WING release check ran the
builder, but the generated artifact was Markdown by default and was not read
back by the release-check schema layer.

## Options considered

1. Keep the capture packet as a human-readable Markdown artifact only.
2. Rely on the in-memory capture-plan schema check.
3. Generate the packet as JSON in no-WING and validate the generated artifact.

## Decision

Use option 3. The release check now writes
`read_surface_physical_readback_capture_packet.json` and validates it immediately
after the builder command succeeds.

## Why this won

The packet is an operator gate at the read-surface safety boundary. Validating
the actual generated file catches drift in top-level authority flags, next
target/action consistency, required physical fixture contracts, blockers,
forbidden actions, and validation commands before later release checks can pass.

## Rejected alternatives

- Markdown-only output is not a stable machine-verifiable release artifact.
- Plan-only validation does not prove the release run wrote a safe packet.
- Acceptance-template or fixture-closure checks happen too late for a bad
  capture packet.

## Implementation plan

1. Switch no-WING capture packet output to JSON.
2. Add a generated-artifact schema validator.
3. Fail closed on live/query/mutation/execution/autonomy claims.
4. Add JSON/Markdown summary output for the artifact gate.
5. Cover valid, missing, unsafe, ordering, and summary cases in tests.

## Test plan

- `python3 -m py_compile tools/no_wing_release_check.py tools/build_read_surface_physical_readback_capture_packet.py`
- Focused pytest for capture-packet artifact validation.
- Broader read-surface/no-WING pytest.
- no-WING quick run.
- Memory lint and diff check.
- Full backend pytest.

## Risks and rollback

This does not provide physical readback or live readiness. Operators still need
to capture real artifacts, review them, accept candidates, and make a separate
fixture-closure decision. Rollback is to remove the validator, summary block,
tests, and docs; no mixer transport path is affected.
