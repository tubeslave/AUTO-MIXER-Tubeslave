# Brief: soundcheck-live-safety-kernel-20260710

## Problem

Current runtime contains live-write paths that are wider than the canonical
readiness matrix: unauthenticated remote WebSocket control, client-controlled
agent auto-apply, headless AutoSoundcheck with live auto-apply, non-systemic
emergency stop, cache-derived write verification and non-transactional rollback.

## Required outcome

- Safe, usable WING soundcheck analysis and proposal pipeline.
- Exactly one supervised physical write boundary for fader/gain-down.
- Remote WebSocket access is authenticated; local desktop remains usable.
- Every reported apply is physical-readback verified.
- Emergency stop covers every writer.
- M005/M006 remain the hardware/owner promotion gates.
- Full validation is green and the evidence-backed score reaches at least
  90/100; otherwise the real blocker is reported instead of inflating the score.

## Constraints

- Preserve the user's dirty `reports/*` and `rule_trials.jsonl` changes.
- Do not touch secrets or `.env`.
- Do not widen live authority beyond operator-approved fader/gain-down.
- One writer in this checkout; Kimi and reviewers are read-only.
- Real-console writes only through the existing supervised runbook with the
  operator present.

## Acceptance evidence

1. Unauthenticated remote clients are rejected before message dispatch.
2. `force_auto_apply` and headless AutoSoundcheck cannot create WING writes.
3. Emergency stop halts agent, AutoSoundcheck and manual retries.
4. Command cache cannot satisfy post-write physical readback.
5. Unverified writes are never reported as applied.
6. Rollback is transaction-bound and detects intervening changes.
7. Full backend, frontend, E2E and no-WING gates pass.
8. Read-only WING/audio smoke confirms the intended live session without
   widening authority.
