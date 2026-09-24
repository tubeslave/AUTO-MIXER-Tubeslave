# WING lab transport boundary repair

Status: accepted for implementation and offline validation only; no hardware authority.

## Context

The explicit lab session uses the production WingClient but omitted its exclusive
mutation boundary. Fake-only tests allow a nonworking integration to look ready.

## Options considered

1. Disable the WingClient guard: rejected.
2. Switch to an old unguarded client: rejected.
3. Bind lab operations to the existing guard with narrow lifetimes and audit.

## Decision

Implement option 3 subject to independent review. No permissive fallback for
transports without the boundary API. Do not alter production policy or CP05.

## Why this won

It repairs the integration while preserving the transport-level refusal of
unowned writes, including asynchronous trailing-edge sends.

Independent design review (Codex subagent McClintock) accepted this direction
with mandatory queue-drain, cancellation, WAL and recovery-order tests. Kimi
proposal review was attempted but returned HTTP 403 access_terminated_error
(session eaaf322b-c792-4c77-b30e-26bdeeaf5b65); no Kimi approval is claimed.

## Rejected alternatives

Direct OSC, blanket address-prefix permission and forged CP05 validation do not
establish safe execution and are not acceptable shortcuts.

## Implementation plan

Reproduce with real WingClient and an in-memory socket. Add exact operation
capabilities, lifecycle serialization and recoverable failure accounting. Revoke
capabilities on every exit. Preserve failed rollback state and writer ownership.

## Test plan

Focused lab and WingClient safety tests, independent review, then the standard
backend suite. Hardware remains a separate attended stage after required gates.

## Implemented software checkpoint

The real-WingClient regression failed before the fix with
`wing_lab_send_failed:/ch/1/fdr`. The final focused set passes 135 tests,
including 36 real guarded-send integration cases with socket creation and DNS
blocked. The independent review is recorded in
`.ai/reviews/wing-lab-transport-boundary-20260924.md`.

The lab still restores all corrections on stop. This patch does not convert it
into a persistent soundcheck workflow and does not repair/rebind CP05. No console
write was performed during this software checkpoint.

Full-suite repeat after lease release: 7568 passed, 3 skipped, 1 failed at the
dynamic CP05 tracked-tree gate. Release/hardware readiness remains unproven.
See `Docs/reports/WING_LAB_TRANSPORT_REPAIR_20260924.md` for exact evidence.

## Risks and rollback

Queued writes, expiry during a batch, concurrent human changes, failed journal
writes and partial rollback must not produce broad or lingering authority.
The lab stays disabled by default. Reverting this focused patch restores the
previous fail-closed transport behavior; do not revert unrelated user changes.
