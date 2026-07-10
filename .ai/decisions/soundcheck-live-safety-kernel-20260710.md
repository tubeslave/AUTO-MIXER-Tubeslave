# Decision: soundcheck live safety kernel

Adopt the incremental single-authority plan. The current supervised manual gate
becomes the only executable WING write path in this checkpoint; AutoSoundcheck
and AI agent are forced proposal-only. Remote binds require authentication,
post-write success requires physical readback, emergency stop covers all writers,
and rollback becomes transaction-bound. A dedicated broker extraction follows
only after behavior is proven.
