# ADR — M116 M110 HW01 No-Signal Truth Sync

## 1. Context

Canonical state said no physical WING existed and HW01 had never started. An
operator-attended M115 route-only run instead reached WING Rack serial
`0100Q1V0604AAE`, firmware `3.0.5-4-ga4ea833d:release`, and a real CoreAudio
`WING` stream at 48x48/48 kHz. It failed closed before route-artifact issuance
with `master_channel_signal_missing`. No reset, correction or live write ran.

## 2. Options considered

Leave the stale state, promote HW01, import the local summary as accepted proof,
add a new product artifact schema, or record a truth-only diagnostic checkpoint.

## 3. Decision

Record `M110-HW01` as attempted and blocked on missing master signal. Preserve
the runtime summary/log path and digest as ephemeral diagnostic evidence, keep
serial-bound USB identity and accepted route proof false, append rather than
rewrite the ledger, and synchronize all current operator surfaces.

## 4. Why this won

It is the smallest change that makes operator guidance true while preserving
the exact safety boundary that prevented silent input from authorizing reset or
mix corrections.

## 5. Rejected alternatives

HW01 promotion lacks an issued artifact. A new runner schema is useful future
work but not needed to correct current truth. Rewriting M113 history would
destroy a correct record of that earlier no-device observation.

## 6. Implementation plan

Update state, append the M116 ledger row, synchronize current docs/project
mirrors, regenerate progress, update rails and commit all non-marker changes.
Then compute and commit only the two CP05 markers.

## 7. Test plan

Run JSON/JSONL validation, focused route/capture/runner suites, all WING truth
rails, progress generation tests, full backend tests, frontend validation,
no-WING, memory lint, diff check and independent review. Verify CP05 on the
clean marker-only final HEAD.

## 8. Risks and rollback

Main risks are overclaiming USB identity, accepted evidence or readiness, and
invalidating CP05 after marker rebind. Roll back the M116 truth commit and its
marker-only rebind together. Never use the diagnostic summary as route
authority.
