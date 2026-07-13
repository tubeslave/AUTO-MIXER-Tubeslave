# ADR: decompose soundcheck rollback boundaries and publish mix readiness

Status: accepted and implemented locally on 2026-07-12. Physical promotion is
not implied; normal live authority remains empty. Verification and remaining
gates are recorded in
`Docs/reports/SOUNDCHECK_PLAN_DECOMPOSITION_20260712.md`.

## 1. Context

The supervised soundcheck planner currently places every normal correction in
one transaction. The queue, server and coordinator validate the entire plan
before the first write. In the archived M006 plan, 15 local transition issues
therefore suppress all 51 operations, including 36 that pass current local
guards. Separately, engine and frontend lifecycle states can imply completion
without proving that musically required corrections were resolved.

Standing normal live authority is zero. This decision must remove the
architecture/truth defect without promoting any WING capability.

## 2. Options considered

1. Teach the coordinator to skip blocked operations.
2. Add a full stateful stage orchestrator immediately.
3. Emit one child transaction per rollback-coupled semantic operation under an
   immutable parent manifest, plus a separate readiness contract.
4. Patch only the completion wording/UI.

## 3. Decision

Choose option 3.

- `SoundcheckTransactionPlanBundle` becomes a versioned parent manifest with
  unique run identity, stable content fingerprint and plan revision.
- Normal and separate-approval work is emitted as deterministic child plans,
  one semantic operation per child by default.
- Every operation carries stage, stage order, logical correction ID and
  `required_for` metadata.
- The existing coordinator remains the atomic fail-closed execution kernel.
- Typed assessment/readiness models distinguish global hazards, local residuals,
  already-satisfied state, split/replan requests, operator review and forbidden
  corrections.
- Proposal-only derivation always reports `soundcheck_incomplete` unless a
  separate trusted baseline proof establishes readiness.
- Generic gain/fader proposal rows are suppressed when their supervised child
  plan is present, avoiding two apply surfaces for one correction.
- Large conservative moves are advanced through server-generated bounded
  revisions. Multi-field compressor steps advance all constrained fields from
  one fresh snapshot and carry a hard anti-loop budget.
- A trusted `global_fader_translation_v1` alternative may replace a fader boost
  only when every selected source has a fresh absolute fader target, source and
  Main evidence are fresh, and the complete target vector can be shifted down.
  It preserves relative balance, never raises a fader and still requires a new
  operator approval. Other substitutions remain operator-review-only.
- Per-source/per-stage coverage and typed derivation residuals prevent skipped,
  disabled, rejected or failed channel corrections from disappearing.
- Every child approval refreshes the current operation and the relevant prior
  stage checkpoints through two consistent physical readback passes. Final
  readiness uses two full passes, per-operation timestamps, a fresh real audio
  callback with sample-count progression, current console serial/firmware, and
  current `/io/out/USB/47|48` Main L/R readback.
- Parent identity and child outcomes are fsync-journaled. Restart restores the
  incomplete run visibly and requires fresh preflight/reimport/reconciliation.
- `safe_core_ready` and `show_mix_ready` are terminal evidence states;
  `soundcheck_incomplete` remains mandatory while any musically required
  residual is unresolved.
- Constructive fader/send/gain-up authority remains disabled.
- The v2 parent manifest carries a canonical content digest over musical
  targets, group structure, source coverage and readiness residuals. First
  trusted import and journal recovery reject nested-content tampering.
- A selected-source fader transaction binds the exact analysis deadline at
  coordinator entry. A missing, extended or expired deadline revokes authority.
- Final source-path reconciliation participates in each fader operation's
  checkpoint; path loss also blocks later same-source stages.
- A blocked result with any possible console send is terminal and retains its
  idempotency reservation until operator reconciliation.

## 4. Why this won

It fixes the wrong rollback boundary directly while preserving the proven
single-operation coordinator, WAL, CAS readback and rollback kernel. Durable
parent state, strict revision identity, explicit operator acceptance and fresh
stage/final checkpoints make bounded continuation recoverable without granting
background authority. Truthful readiness prevents partial success from being
presented as a completed mix.

## 5. Rejected alternatives

- Coordinator skip behavior was rejected because it would weaken atomicity and
  hide dependency failures.
- A broad autonomous orchestrator was rejected. The accepted implementation is
  limited to journaled identity/outcomes and explicit server-generated next
  revisions; it cannot approve or apply a child by itself.
- Status-only wording was rejected because the `0/51` plan defect would remain.
- Immediate constructive moves remain rejected. Current route and audio proof
  are now fresh, but gain-up/send-up/fader-up still require a separately proven
  constructive lane and explicit policy promotion.

## 6. Implementation plan

1. Add stage/run/revision/fingerprint metadata and child-plan decomposition.
2. Add a pure soundcheck readiness/assessment contract.
3. Integrate proposal-only incomplete readiness into engine/status surfaces.
4. Suppress duplicate generic apply rows when child plans exist.
5. Render mix readiness independently from runtime `running`/derivation state.
6. Add exact regression, focused integration and no-authority tests.
7. Update project truth and re-score using the brief rubric.
8. Add durable parent recovery, bounded replan acceptance and explicit
   read-only finalization.
9. Add per-source coverage, two-pass consistency, current Main-route identity,
   fresh real-input progression and approval-expiry gates.

## 7. Test plan

- Deterministic child ordering under reversed input.
- One local child blocker cannot erase a sibling child.
- Global hazards remain fail-closed in existing server/coordinator tests.
- M006-shaped offline assessment retains all 51 logical corrections and exact
  residual reasons.
- Required unresolved corrections prevent ready states.
- Disabled processors cannot resolve an EQ/compressor target as audible.
- A signalled but unidentified/auto-disabled source cannot claim stage coverage.
- Multi-constraint compressor splitting converges without oscillation.
- Trusted global fader translation preserves relative targets and is down-only.
- Stale, slow or mid-sweep-changing snapshots fail both child and final gates.
- Fully already-satisfied and zero-correction plans have an explicit read-only
  finalization path.
- Restarted parent runs remain visible and incomplete.
- Claimed approval expiry and writer contention block before any physical query.
- Rolled-back operations never count as applied merely because their ID appears
  in an intermediate result.
- Bridge creates only one apply surface per logical correction.
- Full backend/frontend/E2E/no-WING regression remains green.

## 8. Risks and rollback

The main compatibility risk is a larger proposal count. The UI consumes the
parent manifest as the aggregate surface; child plans remain bounded and
deterministic. If decomposition causes an integration regression, the source
change can be reverted without touching the coordinator, family registry or
historical evidence. No rollback may widen live authority.
