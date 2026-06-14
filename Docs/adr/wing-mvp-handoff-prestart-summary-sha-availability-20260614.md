# ADR: WING MVP handoff prestart summary SHA availability

## Context

The no-WING handoff packet binds a prestart summary by SHA256 when the summary
file exists. It also exposes `backend_port_prestart_summary_sha256_available`.
Before this ADR, no-WING handoff validation did not require that availability
flag to agree with the digest field.

## Options considered

1. Leave digest availability as informational.
2. Remove the availability flag.
3. Validate availability against digest presence and fail closed on mismatch.

## Decision

Use option 3.

## Why this won

The no-WING handoff summary is the operator-visible bridge into the WING
hardware step. It should not carry contradictory digest state. A digest without
availability or availability without a digest is treated as an invalid handoff
artifact.

## Rejected alternatives

Option 1 was weaker because contradictory packet state could pass validation.
Option 2 removed useful placeholder state from offline handoff previews.

## Implementation plan

- Compute packet digest availability from `backend_port_prestart_summary_sha256`.
- Compare it to `backend_port_prestart_summary_sha256_available`.
- Add a dedicated validation violation on mismatch.
- Render the availability and match flags in no-WING summary output.

## Test plan

- Focused no-WING validator tests for matching and mismatched availability.
- Focused Project Director and WING MVP rail tests after docs update.
- no-WING quick gate.
- Full pytest before commit.

## Risks and rollback

The change is read-only and offline. Rollback is reverting the availability
comparison and summary fields if a legitimate no-WING generated packet is
unexpectedly rejected.
