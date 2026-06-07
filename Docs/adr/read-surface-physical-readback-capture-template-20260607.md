# ADR: Read-Surface Physical Readback Capture Template

## Context

The read-surface physical readback chain currently generates a capture packet,
artifact review queue, acceptance template, and fixture-closure plan. The packet
now lists all remaining targets, but there was no pending artifact template for
the operator to fill before the review queue intake.

## Options considered

1. Leave the packet and queue as the only operator handoff.
2. Add pending artifact fields to the capture packet.
3. Add a separate capture-template artifact and no-WING gate.

## Decision

Use option 3. Add a dedicated pending capture-template artifact between the
capture packet and artifact-review queue.

## Why this won

A separate template keeps the packet focused on target selection and makes the
operator-filled artifact shape explicit. The no-WING validator can then reject
both unsafe authority claims and already filled proof payloads that are still
using the pending-template schema.

## Rejected alternatives

- Packet-only output leaves the operator to infer the required artifact fields.
- Adding pending fields to the packet blurs target handoff and artifact capture.
- Waiting for real hardware proof keeps the next operator step underspecified.

## Implementation plan

1. Build pending capture records from `remaining_capture_targets`.
2. Keep placeholder values for operator-edited fields.
3. Keep execution, candidate, fixture-closure, live-authority, and autonomous
   flags false on the template and every record.
4. Add no-WING artifact validation and summary output.
5. Insert the command before artifact-review queue generation.

## Test plan

- `python3 -m py_compile tools/build_read_surface_physical_readback_capture_template.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_read_surface_physical_readback_capture_template.py`
- Focused no-WING artifact, order, run-release, and summary tests.
- Broader read-surface/no-WING pytest.
- no-WING quick run.
- Memory lint and diff check.
- Full backend pytest if the release-check change remains broad.

## Risks and rollback

This change does not query hardware, accept artifacts, update fixtures, or widen
live authority. Rollback is to remove the new tool, no-WING command and
validator, focused tests, and docs row; no mixer transport path is affected.
