# Proposal: cache-freshness-read-dto-20260605

## Short thesis

Add timestamp-backed cache freshness metadata to read-only Product Layer DTOs,
without changing mixer transport or live apply policy.

## What I understood

The goal requires autonomous console control, but cached state alone is not
safe evidence. A stale fader, EQ, or mute value can produce a wrong decision.
The next safe step is to make cache age explicit before broadening any write
surface.

## Proposed solution

Add `state_timestamps` to WING and dLive clients and update it whenever the
local state cache changes. Add Product Layer helpers that inspect those
timestamps and publish `cache_freshness` with `fresh`, `stale`, `partial`,
`unknown`, or `missing` status. Wire this into channel inventory, channel
processing, and dLive snapshots first.

## Likely files to touch

- `backend/wing_client.py`
- `backend/dlive_client.py`
- `backend/operator_product_state.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`

## Alternatives considered

- Query hardware from Product Layer builders to refresh stale values. Rejected
  because builders must remain display-only and transport-free.
- Treat any cached value as fresh. Rejected because it weakens live safety and
  would support false readiness claims.
- Add freshness to every console surface in one pass. Rejected for this cycle
  because channel/control and dLive parity are the highest leverage starting
  points.

## Risks

- Existing runtime caches without timestamps will report `unknown`, which may
  surface more incomplete status in the UI.
- Top-level freshness does not yet provide per-field proof for every proposed
  action.

## Test plan

- Unit tests for fresh channel cache, stale processing cache, and missing dLive
  cache.
- Focused Product Layer and inventory tests.
- Full pytest suite and diff check.

## Where the other agent may disagree

Kimi may argue for physical fixture capture first. My counterpoint is that
fixture capture needs a stable freshness contract, and this change keeps
physical proof clearly separate from cached freshness.
