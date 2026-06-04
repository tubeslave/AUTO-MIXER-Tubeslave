# ADR: Autonomy Promotion Readiness Snapshot

Date: 2026-06-04

## 1. Context

The project goal requires moving from analysis and assisted proposals toward
supervised and eventually fully autonomous console control. Proposal coverage is
now broad, but proof coverage is not. The live executor remains limited to the
Product Layer supervised fader/gain gate, and Full Autonomous mode remains
blocked.

## 2. Options considered

- Promote the next action family directly.
- Keep readiness spread across existing inventories.
- Add a read-only promotion readiness snapshot.

## 3. Decision

Add `autonomy_promotion_readiness` as a read-only Product Layer snapshot. It
combines capability rows, safety proof rows, write-surface policy, and
read-surface coverage into fail-closed promotion rows. It does not add an
executor, allowed write kind, or live apply path.

## 4. Why this won

This makes the path to autonomy measurable without overstating readiness.
Fader can be shown as supervised-only; gain can be shown as executor-supported
but safety-proof-blocked; broad console surfaces can be shown as blocked by
missing safety coverage, read coverage, or manual-only policy.

## 5. Rejected Alternatives

- Direct promotion was rejected because gain, EQ, dynamics, gates, FX, routing,
  and scenes still lack required proof.
- Docs-only readiness was rejected because tests should enforce fail-closed
  invariants.
- Treating proposal coverage as readiness was rejected because proposal payloads
  are non-queueable and not actuator proof.

## 6. Implementation Plan

1. Add a pure Python readiness snapshot builder.
2. Cover all major live-control action families.
3. Expose the snapshot through the processing registry.
4. Update the project-goal matrix next step to proof-gated promotion.
5. Add tests for fail-closed status, fader supervised-only readiness, gain proof
   mismatch, and blocked broad surfaces.

## 7. Test Plan

- Compile touched Python modules.
- Run focused readiness and registry tests.
- Run broader proposal-boundary tests.
- Run token and whitespace scans.
- Smoke-check that `autonomous_ready_target_ids` remains empty.

## 8. Risks And Rollback

Risk: readiness rows are mistaken for permission. The snapshot explicitly sets
`live_write=false`, `live_widening_allowed=false`, and
`full_autonomous_ready=false`; row blockers include proof gaps and the blocked
Full Autonomous mode. Rollback is to remove the snapshot wiring; no live path is
introduced.
