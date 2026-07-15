# M122 route artifact SHA prefix

## Problem

The persisted M110 route artifact API correctly exposes a bare 64-character
SHA-256 digest, but the process-private authority copied that bare value into a
field whose downstream M110 bindings use the canonical `sha256:<hex>` form.
The mismatch can reject an otherwise valid same-process attended-control
binding.

## Scope

- Keep the persisted/public artifact object unchanged.
- Canonicalize only `MasterCaptureRouteAuthority.artifact_sha256` at issuance.
- Prove both representations from the exact persisted bytes.
- Do not contact WING, grant live authority, or change routing/capture logic.

## Validation

Run the fixed-route proof test and the adjacent M110 route/control suites, then
review the exact diff before commit.
