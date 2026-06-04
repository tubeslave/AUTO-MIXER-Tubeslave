# ADR: Performance Adaptation Proposal Layer

Date: 2026-06-04

## 1. Context

The project goal requires adaptation to changing musician levels, concert-time
mix adaptation, and real-time corrections. Existing fader/gain and agentic
surfaces can produce shadow information, but there was no Assisted-mode contract
that describes when a live ride candidate is explainable enough for operator
review. The live executor remains limited to supervised fader/gain.

## 2. Options considered

- Keep adaptation rows shadow-only.
- Add a proposal-only adaptation evidence layer.
- Promote live gain/fader rides directly.

## 3. Decision

Add `performance_adaptation_proposal` as a transport-free proposal-only method.
Move `musician_level_adaptation`, `live_adaptation`, and `realtime_corrections`
to `proposal_only` with an Assisted-mode ceiling. The output is non-queueable and
does not create a live write request.

## 4. Why this won

This advances the real autonomous-mixing target while preserving safety. The
method requires baseline level, current readback, rollback state, operator
intent, trend persistence, cooldown, latency budget, headroom, feedback risk,
bleed risk, and confidence evidence before emitting a proposal.

## 5. Rejected alternatives

- Direct live ride promotion was rejected because adaptation decisions need
  replay comparison against engineer actions and hardware proof.
- Shadow-only status was rejected because it leaves no concrete operator-review
  path for live adaptation.
- Queueable fader/gain proposals were rejected for this slice because it would
  blur proposal evidence with supervised execution before the replay gate exists.

## 6. Implementation plan

1. Add a pure Python performance adaptation analyzer.
2. Emit only non-queueable proposal payloads with no `requested_change`.
3. Register the method in processing metadata and proposal contracts.
4. Update the relevant project-goal capability rows.
5. Add tests for safe downward proposals, upward blockers, missing evidence, and
   live-boundary invariants.

## 7. Test plan

- Compile touched Python modules.
- Run focused adaptation tests.
- Run broader proposal-boundary tests for registry and capability matrix.
- Run `git diff --check` and payload smoke checks.

## 8. Risks and rollback

Risk: proposal output could be mistaken for apply-ready automation. Mitigation:
payloads are non-queueable, have `can_apply_live=False`, include unsupported
value-type blockers, and contain no requested change. Rollback is to remove the
method from registry/profile/matrix wiring; no direct live path is introduced.
