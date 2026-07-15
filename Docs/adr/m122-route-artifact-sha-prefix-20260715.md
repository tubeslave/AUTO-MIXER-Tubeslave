# ADR: M122 canonical route artifact SHA inside process-private authority

Status: accepted for implementation on 2026-07-15

## 1. Context

`write_master_capture_route_artifact()` returns a bare hexadecimal SHA-256 for
the persisted audit artifact. That is the correct on-disk/public result shape.
M110 process-private authorities and their downstream receipt bindings,
however, use prefixed `sha256:<hex>` identifiers. Copying the bare digest into
`MasterCaptureRouteAuthority.artifact_sha256` creates a representation mismatch
without any byte mismatch.

## 2. Options considered

1. Change the persisted artifact API to return a prefixed digest.
2. Make every downstream comparison accept both forms.
3. Canonicalize only at the boundary that constructs the process-private
   authority.

## 3. Decision

Choose option 3. Keep `result["artifact"]["sha256"]` as the bare digest of the
exact persisted bytes and store the same digest in canonical prefixed form only
inside `MasterCaptureRouteAuthority`.

## 4. Why this won

The change preserves the established artifact schema and fixes the one boundary
whose consumers already require prefixed identifiers. It avoids dual-form
comparisons throughout the attended-control and receipt chain.

## 5. Rejected alternatives

- Changing the artifact schema would invalidate existing evidence consumers.
- Accepting both forms downstream would spread ambiguity across safety checks.

## 6. Implementation plan

Use `_m110_prefixed_sha256()` when constructing the route authority and extend
the fixed-route test to hash the persisted bytes and assert both representations.

## 7. Test plan

- The persisted result is the exact bare SHA-256 of the artifact file.
- The verified process-private authority exposes `sha256:<same digest>`.
- Existing invalidation, route-capture, and no-write assertions remain green.

## 8. Risks and rollback

Risk is limited to consumers that incorrectly depended on a bare internal
authority digest; current downstream contracts require the prefixed form.
Rollback reverts this commit without changing any persisted artifact bytes.
