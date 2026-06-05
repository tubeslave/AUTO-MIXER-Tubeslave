# dlive-state-physical-readback-artifact-intake-20260605

## Context

dLive state already has a physical readback schema contract for cached dLive
state, passive MIDI/TCP state counts, and WING parity mapping. The next gap is
a read-only intake surface for separately operator-captured JSON artifacts.

dLive MIDI/TCP sends and dLive state writes are live-console operations. Intake
must not be interpreted as permission to send MIDI/TCP, write dLive state,
apply dLive changes, or close physical fixture gaps automatically.

## Goal

Add a dLive-state artifact-intake layer as manual-review evidence only:

- accept valid operator-captured artifacts as review candidates;
- reject invalid or mixed valid+invalid artifact sets;
- preserve `fixture_closure_allowed_by_intake=false`;
- preserve `midi_send_allowed=false`, `tcp_send_allowed=false`,
  `dlive_write_allowed=false`, and `dlive_apply_allowed=false`;
- keep live authority and Full Autonomous readiness false.

## Non-goals

- Do not contact dLive or WING hardware.
- Do not send MIDI/TCP messages.
- Do not write or apply dLive state.
- Do not close physical fixture gaps from intake alone.
- Do not add a new executor or live-write path.

## Safety constraints

- Intake must remain transport-free and read-only.
- Any source claiming MIDI/TCP/dLive write authority must fail validation or be
  normalized back to false in summaries.
- Mixed candidate sets must fail closed when any invalid artifact is present.
- Product Layer and frontend rows must render intake as read-only evidence.

## Expected evidence

- Focused backend tests for dLive-state intake.
- no-WING release-check tests for the read-only gate.
- Product Layer and frontend adapter contract tests for read-only visibility.
- Memory lint and diff hygiene.
