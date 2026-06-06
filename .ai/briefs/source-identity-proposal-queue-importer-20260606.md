# Brief: source-identity-proposal-queue-importer-20260606

## Context

The source identity queue-ordering plan can name future queue-import candidates,
but it still cannot produce import specs or exercise the native
`operator_proposal_queue` boundary.

## Objective

Add an operator-gated source identity queue importer that consumes
queue-ordering plan records plus source proposal payloads, emits dry-run import
specs by default, and mutates only an explicit target queue when invoked with
operator acknowledgement.

## Safety Constraints

- Do not touch WING, dLive, OSC, MIDI, TCP, sockets, or mixer clients.
- Product Layer preview must call the importer with `apply=false`.
- Do not mutate `operator_proposal_queue` unless a caller passes an explicit
  target queue with `apply=true` and `operator_ack=true`.
- Do not mutate Product Layer proposal order.
- Do not mutate runtime memory, classifier models, source identity trust stores,
  downstream priority state, or live mixer state.
- Queue import is not live apply authority; supervised console apply still goes
  through existing operator proposal approval and manual write gates.

## Expected Evidence

- A backend queue importer module validates queue-ordering plan envelope,
  summary counts, plan records, duplicate IDs, payload links, UTC timestamps,
  and safety flags.
- Valid plan records plus source proposal payloads produce dry-run import specs.
- Missing plan records or missing source payloads stay not-ready/source-payload
  required.
- `apply=true` without `operator_ack=true` is blocked.
- `apply=true` and ack without an explicit target queue is blocked.
- Applying with ack mutates only the caller-provided `OperatorProposalQueue`.
- Product Layer, registry, and capability matrix expose dry-run import evidence
  without live authority.
