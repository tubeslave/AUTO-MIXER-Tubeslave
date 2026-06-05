# Proposal: recommendation-bridge-cache-freshness-gate-20260605

## Short thesis

Apply the same cache freshness promotion rule to SafeGain and soundcheck
recommendation imports that now protects operator-analysis proposals.

## What I understood

The Automixer goal depends on SafeGain and soundcheck recommendations becoming
safe supervised actions. That path should not trust stale runtime console cache
for current trim or fader values.

## Proposed solution

Add read-only cache freshness checks in
`backend/operator_recommendation_bridge.py`. When a runtime mixer client exposes
`state`, the bridge checks the relevant fader/gain address and
`state_timestamps`. Fresh cache adds safety evidence. Stale, unknown, or missing
cache strips `requested_change`, changes the proposal kind to `analysis`, and
stores blocker metadata. Update soundcheck import in `server.py` to pass the
runtime mixer client.

## Likely files to touch

- `backend/operator_recommendation_bridge.py`
- `backend/server.py`
- `tests/test_operator_product_state.py`

## Alternatives considered

- Query the console from the bridge. Rejected because the bridge is a
  write-free, transport-free adapter.
- Treat soundcheck current_state as enough. Rejected for import-to-queue because
  apply safety needs current console state evidence too.
- Block pure adapter tests without runtime state. Rejected because those tests
  validate conversion artifacts, not queue import safety.

## Risks

- Runtime imports may produce more analysis-only recommendations when timestamps
  are absent.
- Cache freshness is still not physical readback proof; supervised apply must
  continue to pre-read and verify.

## Test plan

- Focused bridge tests for fresh SafeGain, stale SafeGain, and stale soundcheck
  cache.
- Broad operator/server/readiness tests.
- Full pytest suite and diff check.

## Where the other agent may disagree

Kimi may prefer making no-runtime-cache proposals non-applyable everywhere. My
counterpoint is that pure conversion uses should stay usable; queue imports now
receive the runtime mixer client and enforce the stricter gate.
