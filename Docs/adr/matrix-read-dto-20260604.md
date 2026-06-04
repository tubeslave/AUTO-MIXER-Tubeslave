# ADR: Matrix Read DTO

Date: 2026-06-04

## 1. Context

Full autonomous mixing requires trusted matrix visibility before master/output
optimization can be promoted. Matrix state was previously represented only as
raw cache or insert helper behavior, not as a Product Layer contract.

## 2. Options considered

- Leave matrix state as raw cache.
- Call active insert query helpers from Product Layer.
- Add a cached read-only matrix snapshot.

## 3. Decision

Add `matrix_snapshot`, built only from runtime cached state. It exposes matrix
basic controls, EQ bands, and pre/post insert state with explicit section
coverage.

## 4. Why this won

It advances console observability without contacting hardware or widening live
authority. It also creates a stable schema for later matrix send-state and
physical-read fixture work.

## 5. Rejected Alternatives

Raw cache was rejected because consumers cannot infer completeness safely.
Active query helpers were rejected because Product Layer snapshots must remain
display/read-only. Matrix apply paths were rejected because readback, rollback,
and proof gates are incomplete.

## 6. Implementation Plan

1. Add cached matrix snapshot helpers.
2. Expose `get_matrix_snapshot`.
3. Update read-surface inventory with current coverage and blockers.
4. Add tests for full cache, missing cache, display-only safety, and handler
   round-trip.

## 7. Test Plan

- Compile touched Python files.
- Run focused operator product-state and read-surface tests.
- Run broader handler, registry, readiness, and mode tests.
- Run the standard full pytest command.

## 8. Risks and Rollback

Risk: cached coverage is mistaken for physical proof or live authority. The DTO
sets `queries_console=false`, `live_write=false`, and
`full_autonomous_ready=false`, and inventory remains partial. Rollback is to
remove the snapshot builder, handler, and tests; no live write path is
introduced.
