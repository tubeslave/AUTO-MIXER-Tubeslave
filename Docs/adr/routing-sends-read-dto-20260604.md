# ADR: Routing Sends Read DTO

Date: 2026-06-04

## 1. Context

Full autonomous mixing requires route and send visibility before routing, FX
sends, monitor sends, or output patching can be trusted. Routing writes are
high-risk because they can repatch broad console state.

## 2. Options considered

- Keep only static Dante routing prose.
- Call active WING routing query helpers.
- Add a static-plus-cached read-only routing/sends snapshot.

## 3. Decision

Add `routing_sends_snapshot`, built from static Dante config and runtime cached
state only. It exposes channel bus sends, main sends, and output route source
fields with explicit coverage.

## 4. Why this won

It advances routing observability without contacting hardware or widening live
authority. It also gives future route/send diff and physical-query fixtures a
stable schema.

## 5. Rejected Alternatives

Static prose alone was rejected because it does not expose runtime sends or
output source cache. Active query helpers were rejected because Product Layer
snapshots must not query hardware. Queueable routing/send proposals were
rejected because diff, readback, rollback, and proof gates are incomplete.

## 6. Implementation Plan

1. Add cached/static routing-send snapshot helpers.
2. Expose `get_routing_sends_snapshot`.
3. Update read-surface inventory with current coverage and blockers.
4. Add tests for cached route/send rows, missing cache, display-only safety,
   and handler round-trip.

## 7. Test Plan

- Compile touched Python files.
- Run focused operator product-state and read-surface tests.
- Run broader handler, registry, readiness, and mode tests.
- Run the standard full pytest command.

## 8. Risks and Rollback

Risk: cached routing coverage is mistaken for physical proof or live authority.
The DTO sets `queries_console=false`, `live_write=false`, and
`full_autonomous_ready=false`, and inventory remains partial. Rollback is to
remove the snapshot builder, handler, and tests; no live write path is
introduced.
