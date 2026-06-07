# Proposal: Read-Surface Capture Packet Target Manifest

## 1. Short thesis

Extend the capture packet into a full remaining-target manifest while keeping it
read-only and non-executing.

## 2. What I understood

The current no-WING chain proves the generated capture packet, review queue,
acceptance template, and closure plan as safe artifacts. The capture packet,
however, only describes the next target in detail. Since full console autonomy
requires every read surface to be physically proven, the operator packet should
also carry a complete list of remaining capture targets.

## 3. Proposed solution

- Reuse `read_surface_remediation_plan` actions to build
  `remaining_capture_targets`.
- Include surface id, remediation action id, target console surface, read mode,
  contract id, expected artifacts, blockers, and fail-closed safety flags.
- Validate the generated packet in no-WING for target count, sequence, first
  target, contract coverage, and false target-level authority flags.
- Add target ids to summary JSON and Markdown.

## 4. Likely files to touch

- `tools/build_read_surface_physical_readback_capture_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_capture_packet.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/read-surface-capture-packet-target-manifest-20260607.md`
- `Docs/adr/read-surface-capture-packet-target-manifest-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_CAPTURE_PACKET_TARGET_MANIFEST_20260607.md`
- `Docs/INDEX.md`

## 5. Alternatives considered

- Leave the packet as next-target-only.
- Create a separate target-manifest artifact.
- Make the capture plan generate all capture steps.

## 6. Risks

The main risk is confusing a target manifest with permission to capture, accept,
or close fixtures automatically. The target entries must stay metadata-only and
must not grant query, mutation, live write, or execution authority.

## 7. Test plan

- py_compile for the builder and no-WING checker.
- Focused capture-packet and no-WING artifact tests.
- Broader read-surface/no-WING tests.
- no-WING quick run.
- Memory lint and diff check.

## 8. Where the other agent may disagree

Kimi may prefer a separate manifest artifact to avoid growing the packet. The
reason to keep it inside the packet is operator ergonomics: the next target and
the complete remaining target set belong in the same read-only handoff.
