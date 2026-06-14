# Brief: WING MVP handoff operator sequence cardinality

## Task

Make the no-WING handoff validator fail closed when the operator sequence gains
extra or missing steps around the supervised prestart gate.

## Context

The WING read-only handoff already requires `operator_sequence_count: 10` and a
final supervised prestart gate at index `9`. Before this task, validation could
report these values but did not fail closed on a coherently shifted sequence.

## Constraints

- Keep the change offline and read-only.
- Do not contact WING or the backend.
- Do not grant live authority.
- Preserve the current supervised WING Rack MVP boundary.

## Done

- Handoff validation checks exact operator sequence count.
- Handoff validation checks the supervised prestart gate stays at contract index
  `9`.
- Summary JSON and Markdown expose expected count/index and match flags.
- Focused tests cover a coherently shifted extra-step packet.
