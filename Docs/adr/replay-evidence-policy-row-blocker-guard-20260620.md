# ADR: Replay Evidence Policy Row Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed presentation hardening that prevents operator surfaces
from implying promotion readiness while the integration bundle blocks planning.

Replay Evidence shows a policy row for offline replay runs. The raw
`offline only` and `no live write` text is accurate, but it can still look like
normal policy readiness unless the visible global blocker is reflected there.

## Options Considered

1. Leave the Replay Evidence policy row unchanged.
2. Hide the policy row while the integration bundle is blocked.
3. Keep the row visible and override only its local display values.

## Decision

Use option 3. `ReplayEvidencePanel` derives local `replayPolicyLabel`,
`replayPolicyDetail` and `replayPolicyDetailClassName` from
`integrationBundleApplyBlocked`. While blocked, the Policy row shows
`bundle blocked · no live apply`. Otherwise it preserves the raw replay policy
and live-write detail.

This does not mutate replay run input, decision trace input, task data, backend
status input, backend result input, safety monitor input or readiness input.

## Why This Won

The operator keeps useful replay policy context while the UI makes the global
promotion blocker explicit at the policy point. It is local presentation logic
only and does not alter replay records, backend contracts or apply paths.

## Rejected Alternatives

Leaving the row unchanged keeps a misleading status gap. Hiding the row removes
useful offline evidence and is not required for safety because the row can fail
closed locally.

## Implementation Plan

- Add local replay policy display derivation in `ReplayEvidencePanel`.
- Render the policy row from the local display values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local policy-row status derivation.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Replay Evidence presentation. Rollback removes the local
policy-row derivation, the M061 rails and the M061 docs/state entries. M001
remains blocked on real WING read-only evidence either way.
