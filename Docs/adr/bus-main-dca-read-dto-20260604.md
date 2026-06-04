# ADR: Bus Main DCA Read DTO

Date: 2026-06-04

## 1. Context

Full autonomous mixing requires trusted observability for buses, mains, groups,
and DCAs. The existing group/bus UI snapshot summarizes channels, but it does
not normalize cached console readback for these surfaces.

## 2. Options considered

- Keep using only synthetic group summaries.
- Call active WING query helpers from Product Layer.
- Add a cached read-only bus/main/DCA snapshot.

## 3. Decision

Add `bus_main_dca_snapshot`, built only from runtime cached state. It exposes
bus, main, and DCA rows with explicit section coverage and no query or write
authority.

## 4. Why this won

It improves console observability while preserving the safety boundary. It also
gives future physical-read fixtures and stale-cache timestamps a stable schema.

## 5. Rejected alternatives

Synthetic summaries were rejected because they do not prove console readback.
Active query helpers were rejected in this step because Product Layer snapshots
must not contact hardware. Live bus/main/DCA apply was rejected because proof,
rollback, and readback are incomplete.

## 6. Implementation plan

1. Add cached bus, main, and DCA snapshot helpers.
2. Expose `get_bus_main_dca_snapshot`.
3. Update read-surface inventory current coverage and blockers.
4. Add tests for complete cached coverage, missing cache, display-only safety,
   and handler round-trip.

## 7. Test plan

- Compile touched Python files.
- Run focused operator product-state and read-surface tests.
- Run broader handler, registry, readiness, and mode tests.
- Run the standard full pytest command.

## 8. Risks and rollback

Risk: cached coverage is mistaken for physical proof or live authority. The DTO
sets `queries_console=false`, `live_write=false`, and
`full_autonomous_ready=false`, and the inventory remains partial. Rollback is
to remove the snapshot builder, handler, and tests; no live write path is
introduced.
