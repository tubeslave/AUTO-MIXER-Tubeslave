# ADR: Long-Term Memory Learning Proposal Layer

Date: 2026-06-04

## 1. Context

The project goal requires long-term memory and continuous improvement from
accumulated data, logs, tests, and previous concerts. Existing decision memory,
telemetry export, and replay artifacts provide raw evidence, but
`long_term_memory_learning` was still shadow-only. The live executor remains
limited to supervised fader/gain.

## 2. Options considered

- Keep long-term memory learning shadow-only.
- Add a proposal-only memory-learning review contract.
- Let memory learning write directly to memory or live actions.

## 3. Decision

Add `long_term_memory_learning_proposal` as a transport-free proposal-only
method. Move `long_term_memory_learning` to `proposal_only` with an Assisted
mode ceiling. The output is non-queueable, does not mutate memory, and does not
create a live write request.

## 4. Why this won

This advances the AI Mixing Engineer goal by creating a structured review path
from verified/offline evidence into future memory while preserving the current
live safety boundary. The method records verified decisions, offline replay,
accepted/rejected outcomes, evidence packets, telemetry sessions, engineer
actions, references, genre context, arrangement context, replay correlation,
outcome consistency, conflicts, manual overrides, and safety regressions.

## 5. Rejected Alternatives

- Direct memory writes were rejected because accepted/rejected operator review
  and replay correlation need to be explicit.
- Direct live authority was rejected because learning quality does not prove
  actuator safety.
- Keeping the row shadow-only was rejected because it leaves no operator-review
  contract for learning records.

## 6. Implementation Plan

1. Add a pure Python long-term memory learning analyzer.
2. Emit only non-queueable proposal payloads with `memory_mutation=False`.
3. Register the method in processing metadata and proposal contracts.
4. Update the project-goal capability row.
5. Add tests for safe learning proposals, blockers, and live-boundary
   invariants.

## 7. Test Plan

- Compile touched Python modules.
- Run focused long-term memory learning tests.
- Run broader proposal-boundary tests for registry and capability matrix.
- Run whitespace scan and payload smoke checks.

## 8. Risks And Rollback

Risk: memory-learning proposals are treated as autonomous authority before
operator acceptance and replay validation. The payload is non-queueable, has
`can_apply_live=False`, `memory_mutation=False`, includes unsupported value-type
blockers, and contains no apply request. Rollback is to remove the method from
registry/profile/matrix wiring; no direct live path is introduced.
