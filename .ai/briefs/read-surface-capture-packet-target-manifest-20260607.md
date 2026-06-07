# Brief: Read-Surface Capture Packet Target Manifest

## Task

Make the read-surface physical-readback capture packet expose every remaining
capture target, not only the next target.

## Context

The capture packet already gives an operator the next physical-readback target
and stays non-executing. The broader project goal still needs all console read
surfaces proven from physical readback before live authority can widen. Operators
need the packet to show the whole remaining target set so the handoff can move
toward closing every read-surface gap.

## Safety Boundary

- Do not query or mutate a mixer.
- Do not execute tests from the packet.
- Do not accept artifacts, update fixtures, or close read-surface gaps.
- Keep all target-level execution, live-write, mutation, and query flags false.

## Acceptance Criteria

- Capture packets include `remaining_capture_targets`.
- The target list preserves the physical-gap surface order and puts the next
  target first.
- no-WING validates target counts, action ids, contract ids, and fail-closed
  target flags.
- Summary JSON/Markdown expose target ids for operator review.
- Focused tests cover builder, validator, unsafe target flags, and summary output.
