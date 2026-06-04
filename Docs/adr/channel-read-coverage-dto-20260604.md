# ADR: Channel Read Coverage DTO

Date: 2026-06-04

## 1. Context

Autonomous console decisions need to know whether channel data is actually
available. `build_channel_inventory` already exposes read-only cached state and
capture meters, but downstream code could only infer coverage from `None`
values.

## 2. Options considered

- Keep coverage implicit.
- Add a separate coverage endpoint.
- Add per-channel coverage metadata inside `channel_inventory`.

## 3. Decision

Add a `read_coverage` DTO to each channel row and an aggregate inventory-level
coverage summary. The DTO covers identity, routing, metering, basic controls,
and cached gate/dynamics.

## 4. Why this won

The existing endpoint is already the Product Layer channel contract. Extending
it preserves old fields while making source and completeness explicit for UI,
reviews, and mode gates.

## 5. Rejected alternatives

Implicit `None` checks were rejected because they cannot distinguish missing
readback from valid zero values. A separate endpoint was rejected because it
would duplicate channel inventory assembly and increase drift risk.

## 6. Implementation plan

1. Add helper builders for section and aggregate read coverage.
2. Attach `read_coverage` to every channel inventory row.
3. Attach aggregate `read_coverage` and `summary.read_coverage_status`.
4. Update the read-surface inventory row to reflect current and remaining
   channel-read gaps.
5. Add tests for cached and missing read sections.

## 7. Test plan

- Compile touched Python files.
- Run focused operator product-state and read-surface tests.
- Run broader readiness/mode/registry tests.
- Run the standard full pytest command.

## 8. Risks and rollback

Risk: the DTO is mistaken for full channel readback. The read-surface inventory
stays `partial` and names the remaining blockers. Rollback is to remove the DTO
fields and helper functions; no live path or mixer mutation is introduced.
