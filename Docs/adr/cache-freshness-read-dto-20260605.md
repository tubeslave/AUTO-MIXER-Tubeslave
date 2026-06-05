# ADR: cache-freshness-read-dto-20260605

## Context

The Automixer Product Layer exposes cached console state for channels,
processing, and dLive parity. Cached state can become stale, and stale values
must not be treated as physical proof for autonomous console decisions.

## Options considered

1. Leave cache freshness implicit.
2. Add timestamp-backed freshness metadata to read-only DTOs.
3. Refresh cache by querying mixers from Product Layer builders.

## Decision

Use option 2. Track cache update timestamps in WING and dLive clients and expose
read-only freshness metadata in channel inventory, channel processing, and
dLive snapshots.

## Why this won

It gives the cockpit and future safety gates evidence about cache age without
sending OSC/MIDI or widening any live apply path.

## Rejected alternatives

Option 1 allows stale cache to look equivalent to fresh readback. Option 3
violates the Product Layer boundary by making display DTOs query hardware.

## Implementation plan

- Add `state_timestamps` and local cache update helpers to WING and dLive
  clients.
- Add Product Layer timestamp parsing and freshness summarization.
- Publish `cache_freshness` in channel inventory, channel processing, and dLive
  snapshots.
- Update inventory, focused tests, and council documentation.

## Test plan

- `PYTHONPATH=backend python -m pytest tests/test_operator_product_state.py tests/test_read_surface_inventory.py -q`
- Broader server/registry/readiness tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `git diff --check`

## Risks and rollback

Risk: top-level freshness may be mistaken for per-field physical proof.
Mitigation: status names remain explicit and inventory blockers still require
physical fixtures and per-field freshness evidence. Rollback is removing the
timestamp helpers, Product Layer freshness payloads, tests, and inventory/docs
updates.
