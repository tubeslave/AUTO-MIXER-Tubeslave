# Council fallback critique of Codex proposal

## 1. Strongest part

The incremental plan minimizes risk and keeps the local Electron/read-only
soundcheck path usable while high-risk behaviors are disabled.

## 2. Possible mistakes or blind spots

The existing manual gate cannot be called a true single writer while other
modules still hold raw mixer clients. Tests and a static ownership guard must
make that temporary claim enforceable.

## 3. Where it is fragile

Authentication must be mandatory whenever the bind is non-loopback; a config
combination that exposes an unauthenticated listener would recreate the P0.
Likewise, changing throttle behavior must not let readback run before a deferred
write is actually delivered.

## 4. What should be kept

Proposal-only AutoSoundcheck/agent behavior, physical-only post-write proof,
system emergency generation and transaction-bound rollback.

## 5. Revised recommendation

Land containment and proof semantics first, add explicit tests forbidding raw
AutoSoundcheck/agent writes, then extract the broker after full validation.
