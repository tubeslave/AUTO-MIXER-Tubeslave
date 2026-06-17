# WING Readonly Runner Verifier Artifact Match Gate

## Context

The WING read-only preflight runner validates the detached no-WING artifact
manifest verifier before any backend/WING WebSocket opens. The verifier report
must prove artifact self-consistency, contract cross-surface consistency, and
summary-surface consistency before the runner can proceed to transport.

## Options considered

1. Keep only generic verifier report value mismatch diagnostics.
2. Add dedicated violations for core artifact match failures.

## Decision

Add dedicated pre-WebSocket violations for these verifier artifact match fields:

- `artifact_manifest_self_consistency_match`
- `artifact_manifest_contract_cross_surface_match`
- `artifact_manifest_self_consistency_summary_surface_match`

## Why this won

These fields are the core proof that the no-WING packet artifacts are mutually
consistent. If any of them fail, the operator should see that root cause
directly before transport opens.

## Rejected alternatives

Relying on generic report value mismatch was rejected because it hides artifact
contract drift behind the same diagnostic used for unrelated verifier value
mismatches.

## Implementation plan

- Extend `tools/wing_readonly_preflight_runner.py` with dedicated artifact match
  mismatch violations.
- Add a pre-WebSocket regression test that mocks all three artifact match fields
  as false.

## Test plan

- Focused runner verifier tests.
- Focused WING/no-WING tests.
- Standard full pytest command.

## Risks and rollback

Risk is limited to stricter offline/pre-WebSocket diagnostics. Rollback is to
remove the dedicated violations while preserving the generic verifier failure.
