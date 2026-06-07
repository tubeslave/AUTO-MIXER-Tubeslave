# Brief: Read-Surface Capture Template Contract Skeleton

## Task

Add contract-derived physical readback payload skeletons to the offline pending
capture template.

## Context

The capture-template artifact already tells the operator which surface to
capture and which top-level fields remain pending. It still leaves the actual
`physical_readback_payload` as a generic placeholder string, so an operator has
to infer the exact JSON shape from the target contract.

## Safety Boundary

- Do not query or mutate WING, dLive, OSC, MIDI, TCP, WebSocket, or any console.
- Do not accept the skeleton as physical proof.
- Do not update fixture baselines or close read-surface gaps.
- Keep execution, live authority, fixture acceptance, and full autonomous flags
  false on both the template record and nested payload skeleton.

## Acceptance Criteria

- Each pending capture record includes a `physical_readback_payload_template`
  derived from the record's required physical readback contract.
- The nested payload skeleton includes contract type, schema id, required
  top-level fields, nested required field sets, pending placeholders, and safety
  flags.
- no-WING validation fails if the nested payload skeleton claims live authority,
  fixture candidacy, mutation, or a mismatched contract payload.
- no-WING summary JSON and Markdown expose the payload-template contract ids,
  artifact types, schema ids, and field counts.
