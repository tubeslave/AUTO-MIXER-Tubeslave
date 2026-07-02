# WING Read-Only DSP Health Readback Aliases

## Context

A real WING read-only preflight run captured physical readbacks for EQ,
dynamics and gate health probes on channel 3. The captured artifact queried
read-only alias paths such as `/ch/3/eq/lg`, `/ch/3/dyn/thr` and
`/ch/3/gate/thr`, while the WING fw3.0.5 protocol PDF still lists typed
threshold paths such as `/ch/1/dyn/thr24` and `/ch/1/gate/thr21`.

## Options considered

1. Reject the DSP health artifact because the read-only aliases differ from
   typed protocol-table paths.
2. Treat the observed aliases as apply-capable WING OSC paths.
3. Accept the observed aliases only for explicit read-only DSP health probes.

## Decision

Accept the observed EQ, dynamics and gate aliases only as explicit read-only
health readback value types for the WING read-only preflight runner and review
tools.

## Why this won

The artifact is useful real-console evidence: it contacted the backend,
queried the console, received physical readbacks, and sent no write or proposal
requests. At the same time, a readback response does not prove write safety,
rollback behavior, or apply semantics for EQ, dynamics or gate surfaces.

## Rejected alternatives

Rejecting the artifact would discard real hardware evidence that can improve
the supervised WING MVP intake. Treating the aliases as live-write mappings
would widen authority without the required staged write, readback and rollback
proof.

## Implementation plan

Add explicit opt-in read-only value types for EQ on, EQ mix, low/high EQ gain,
dynamics on/threshold and gate on/threshold. Keep the runner default value
types as `fader,gain,pan`. Do not grant live authority, readiness closure, or
supervised apply from these readbacks.

For smart-DSP DEQ/dynamic-EQ proof work, also add explicit opt-in read-only
gate-slot value types for `gate/mdl`, `gate/ratio`, `gate/att`, `gate/rel`,
`gate/filt`, `gate/g`, `gate/f`, `gate/q` and `gate/mode`. These aliases only
prove that the connected console can return the current gate-slot/DEQ health
state. They do not prove write mapping, external-key routing, staged rollback,
or live-apply authority.

## Test plan

Run focused WING read-only runner, server query and artifact review tests.
Verify the no-WING gate still denies backend/WING contact and live authority.
For gate-slot/DEQ aliases, add runner request-address tests and artifact-review
tests that accept physical readbacks while keeping all write/proposal request
flags false.

## Risks and rollback

Risk: future code could confuse read-only aliases with apply-capable paths.
Rollback by removing the DSP health value types and keeping the real artifact
as review-only evidence. Any EQ, dynamics or gate live writes still require a
separate ADR, verified mapping, supervised write proof, readback and rollback.
