# ADR: Channel Processing Read DTO

Date: 2026-06-04

## 1. Context

Full autonomous mixing needs reliable visibility into channel processing state:
HPF, EQ, compressor, and gate. Existing WING helpers expose cached values, but
the Product Layer did not provide a normalized read contract for cockpit,
readiness, or future proof fixtures.

## 2. Options considered

- Keep processing readback implicit in raw mixer cache.
- Call active WING query helpers from Product Layer.
- Add a cached read-only processing snapshot.

## 3. Decision

Add `channel_processing_snapshot`, built only from runtime cached state and
channel inventory metadata. It exposes per-channel HPF, EQ, compressor, and
gate DTOs with explicit coverage flags.

## 4. Why this won

It advances console observability without contacting or mutating a mixer. It
also creates a stable schema for later physical-query fixtures and stale-cache
timestamps.

## 5. Rejected alternatives

Raw cache was rejected because consumers cannot infer completeness safely.
Active query helpers were rejected for this step because Product Layer snapshots
must remain display/read-only. Immediate EQ/dynamics apply was rejected because
readback, rollback, and proof gates are incomplete.

## 6. Implementation plan

1. Add cached processing snapshot helpers.
2. Expose the snapshot through `get_channel_processing_snapshot`.
3. Update the read-surface inventory row with the new current coverage and
   remaining blockers.
4. Add tests for cached complete coverage, missing coverage, and handler
   round-trip.

## 7. Test plan

- Compile touched Python files.
- Run focused operator product-state and read-surface tests.
- Run broader handler, registry, readiness, and mode tests.
- Run the standard full pytest command.

## 8. Risks and rollback

Risk: cached coverage is mistaken for physical proof. The snapshot sets
`queries_console=false`, `live_write=false`, and `full_autonomous_ready=false`,
and the inventory remains partial. Rollback is to remove the snapshot builder,
handler, and tests; no live write path is introduced.
