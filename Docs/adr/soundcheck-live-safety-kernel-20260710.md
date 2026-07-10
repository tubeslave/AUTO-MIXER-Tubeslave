# ADR: Soundcheck live safety kernel

## 1. Context

The soundcheck pipeline has strong analysis and proposal capabilities, but the
current runtime contains broader write paths than the canonical WING readiness
matrix permits. The real WING and audio session are currently available, so the
software boundary must be corrected before further supervised proof.

## 2. Options considered

1. Preserve broad AutoSoundcheck apply behind configuration.
2. Build a new broker and migrate every controller in one change.
3. Make the existing supervised gate the only executable authority now, prove
   it, then extract a dedicated broker without semantic changes.

## 3. Decision

Choose option 3. Remote WebSocket binds require authentication; local bind is
loopback by default. AutoSoundcheck and AI agents are proposal-only. The manual
supervised gate is limited to operator-approved fader/gain-down, reports applied
only after physical readback, uses a system emergency generation and performs
transaction-bound compare-and-swap rollback.

## 4. Why this won

It closes the active P0/P1 defects with the smallest live-risk diff while
preserving the existing operator proposal and hardware proof workflow.

## 5. Rejected alternatives

Broad apply is rejected because most families lack accepted hardware proof.
Immediate full migration is rejected because it duplicates live execution logic
and makes regressions harder to isolate.

## 6. Implementation plan

1. Network containment and remote authentication.
2. Remove agent and AutoSoundcheck auto-apply bypasses.
3. Add system-wide emergency stop coverage.
4. Correct WING commanded/observed state and throttle semantics.
5. Require physical post-write proof and transaction-bound rollback.
6. Add regression/static ownership tests and run full validation.
7. Run read-only WING/audio smoke, then the existing operator-present M005/M006
   workflow. No broad family promotion occurs in this ADR.

## 7. Test plan

Targeted auth/handler/transport/server tests; WING rails; no-WING quick gate;
full backend suite; frontend test/build/E2E; read-only WING/audio smoke. Hardware
writes remain inside the existing supervised proof runners.

## 8. Risks and rollback

Removing command-cache mutation may reveal missing readback paths. Any failed
proof keeps writes blocked. Rollback is a code revert plus continued
observation-only operation; it never widens authority.
