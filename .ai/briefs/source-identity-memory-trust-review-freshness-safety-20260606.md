# Brief: source-identity-memory-trust-review-freshness-safety-20260606

## Context

Source identity calibration memory trust review can turn indexed append-only
memory records into operator-review candidates when replay context is fresh.
The freshness value itself is an evidence input, so it must also prove that it
is read-only and did not require live writes or console queries.

## Objective

Harden the trust-review contract so forged or unsafe freshness/context evidence
and inconsistent memory-index statuses fail closed before any review candidate
is exposed.

## Safety Constraints

- Do not create mixer clients, sockets, OSC/MIDI/TCP messages, or live writes.
- Do not import trust-review candidates into the decision queue.
- Do not mutate source identity models, runtime memory, downstream priority, or
  console state.
- Treat `fresh` context as usable only when it carries `read_only=true`,
  `live_write=false`, and `queries_console=false`.
- Treat memory-index status drift as replacement-required evidence, not as a
  candidate source.

## Expected Evidence

- Focused trust-review tests cover unsafe freshness flags and inconsistent index
  status with records.
- Product Layer behavior remains read-only because the hardened builder is still
  transport-free and queue-free.
- Memory lint and backend validation remain green.
