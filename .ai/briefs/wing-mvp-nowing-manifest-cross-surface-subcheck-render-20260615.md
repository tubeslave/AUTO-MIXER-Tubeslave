# Brief: no-WING Manifest Cross-Surface Subcheck Render

## Task ID

wing-mvp-nowing-manifest-cross-surface-subcheck-render-20260615

## Context

The active supervised WING Rack MVP blocker remains `WING read-only connection
proof`. Offline no-WING artifacts are operator gates before a real WING
read-only session and must not hide contract drift behind a coarse summary.

The no-WING summary already computed individual manifest cross-surface subcheck
fields for contract id, digest, and digest-match flags. Those fields were not
part of the manifest summary surface contract and were not rendered in the
operator-readable `summary.md`.

## Goal

Make the no-WING primary summary surfaces expose the individual manifest
cross-surface subcheck verdicts.

## Safety boundary

Offline no-WING reporting only. Do not contact WING, query a console, mutate
mixer state, close readiness, or grant live authority.

## Expected validation

- Focused manifest summary/cross-surface tests.
- no-WING quick.
- Project Director guard.
- Standard pytest before commit.
