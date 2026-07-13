# P1: staged soundcheck plan decomposition and honest mix readiness

## Request

The current supervised soundcheck path is safe but can fail to form a mix from
a poor starting scene. One locally blocked correction must not erase unrelated
safe work, and a technically successful transaction must not be reported as a
completed soundcheck while musically important corrections remain unresolved.

The target is a verified local product-readiness score of at least `95/100`
without granting new standing WING authority or weakening any global safety
stop.

## Confirmed symptom

- `build_soundcheck_transaction_plans()` places every non-high-risk operation
  in one `normal` transaction.
- Queue, server and coordinator preflight the whole plan before the first
  write.
- The real M006 normal plan contained 51 operations. Current transition limits
  would locally reject 15: 4 fader boosts, 6 pan moves above 10 percent, 2
  compressor makeup boosts, 1 relative EQ boost, 1 trim cut above 6 dB and 1
  gate-threshold move above 3 dB.
- The present architecture therefore exposes `0/51` as executable instead of
  preserving the 36 operations that pass current local transition rules.

## Required behavior

1. Preserve plan-wide failure for global hazards: disconnected or stale
   physical readback, emergency stop/generation change, writer conflict,
   unavailable durable rollback journal, unknown rollback state, missing or
   stale stereo Main 47/48 evidence, and insufficient master headroom.
2. Classify each correction before execution as one of:
   - `apply`;
   - `split_steps`;
   - `operator_review`;
   - `blocked` with an exact reason.
3. Partition normal work into ordered stages:
   - `foundation`: gain structure, HPF, gate, polarity/delay and coarse balance;
   - `tonal_dynamics`: EQ, compressor, bus/DCA and final balance;
   - `spatial_fx`: pan, sends, returns and FX;
   - high-risk mute/master/routing/scene work stays in separate approvals.
4. Preserve transaction atomicity inside every emitted executable subplan.
   Runtime failure after any send still rolls back that subplan in reverse
   order and can activate emergency stop.
5. Recompute safe alternatives after a local correction is removed:
   - blocked source gain-up may become a bounded fader move or complementary
     competitor attenuation;
   - blocked EQ boost may become evidence-bound complementary cuts;
   - blocked compressor makeup may become a bounded fader rebalance;
   - if no proven substitute exists, retain operator review and incomplete
     status.
6. A constructive lane may be represented in the plan only when fresh stereo
   Main 47/48 signal/headroom, trusted source identity, feedback-clear evidence
   and an accepted bounded policy are present. This task must not silently
   promote that lane to standing live authority. Input/preamp gain-up stays
   stricter because the WING USB source tap is pre-trim.
7. Publish honest result states:
   - `safe_core_ready` only when required foundation coverage is resolved;
   - `show_mix_ready` only when required tonal/dynamics and spatial/FX coverage
     is also resolved;
   - otherwise `soundcheck_incomplete`, with counts and exact unresolved
     reasons.

## Safety invariants

- No new WING contact or writes during implementation and verification.
- No live family is promoted by source code, tests, offline evidence or this
  ADR.
- Current zero standing live authority remains truthful.
- `apply_operator_proposal` remains the sole Product Layer live entrypoint.
- Every executable operation remains an absolute target with physical snapshot,
  verify, audio recheck and compare-and-swap rollback.
- High-risk routing, scenes, mute and master/output actions never join normal
  channel-strip batches.
- Generated historical evidence is not deleted or rewritten.

## Acceptance criteria

- A regression fixture reproduces the real 51-operation M006 shape and proves
  36 local-safe operations survive classification while all 15 local blockers
  remain visible.
- A single local blocker cannot make an unrelated executable subplan vanish.
- A global blocker still stops all stages before any send.
- Alternatives carry provenance from the rejected operation and cannot invent
  a write kind outside current effective authority.
- Completion is impossible while a required musical correction is unresolved.
- Existing rollback, emergency, idempotency and single-writer tests remain
  green.
- Full backend, frontend, E2E and no-WING validation pass before any score claim.

## Score rubric

| Area | Points |
| --- | ---: |
| Global safety and transaction rollback invariants | 25 |
| Local-block isolation and staged execution semantics | 25 |
| Safe-alternative recomputation and musical coverage truth | 20 |
| Honest operator status/evidence contract | 15 |
| Regression, full-suite and no-WING verification | 15 |

The score cannot reach 95 if the M006-shaped regression is absent, local
blockers still erase safe work, or `show_mix_ready` can be true with unresolved
required corrections.

## Dirty worktree boundary

The checkout contains a large uncommitted audit/release tail and generated
evidence. Only files required by this P1 may be changed. No broad staging,
stashing, cleanup or commit is authorized.
