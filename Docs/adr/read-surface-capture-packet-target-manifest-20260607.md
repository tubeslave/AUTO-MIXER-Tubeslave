# ADR: Read-Surface Capture Packet Target Manifest

## Context

The read-surface capture packet is the operator handoff for physical readback
capture. It previously identified only the next target in detail, while the
review queue reported that all read-surface artifacts were still missing.

## Options considered

1. Keep the capture packet focused only on the next target.
2. Add a separate remaining-target manifest artifact.
3. Add `remaining_capture_targets` to the capture packet and validate it in
   no-WING.

## Decision

Use option 3. The capture packet now includes a full remaining-target manifest
derived from remediation-plan actions, and no-WING validates that manifest before
the packet is accepted as safe.

## Why this won

The packet is already the operator-facing handoff for physical readback capture.
Keeping the complete remaining target set in the same artifact makes it clear
that the system is still blocked on all missing read surfaces, while preserving
the next-target workflow.

## Rejected alternatives

- Next-target-only output hides the size of the remaining proof gap.
- A separate manifest adds another artifact without a stronger safety boundary.
- Generating all capture steps in the plan would broaden the planning contract
  more than needed for this slice.

## Implementation plan

1. Build `remaining_capture_targets` from remediation actions.
2. Keep target-level execution, query, mutation, and live-write flags false.
3. Validate target count, sequence, first target, contract coverage, and flags in
   no-WING.
4. Add target ids to summary JSON and Markdown.
5. Cover builder, validator, unsafe target, and summary cases in tests.

## Test plan

- `python3 -m py_compile tools/build_read_surface_physical_readback_capture_packet.py tools/no_wing_release_check.py`
- Focused capture-packet/no-WING pytest.
- Broader read-surface/no-WING pytest.
- no-WING quick run.
- Memory lint and diff check.

## Risks and rollback

This change does not query hardware, accept artifacts, update fixtures, or widen
live authority. Rollback is to remove the manifest fields, validator checks,
tests, and docs; no mixer transport path is affected.
