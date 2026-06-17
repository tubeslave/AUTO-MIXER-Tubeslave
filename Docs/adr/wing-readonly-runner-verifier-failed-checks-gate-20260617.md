# WING Readonly Runner Verifier Failed Checks Gate

## Context

The WING read-only preflight runner validates the detached no-WING artifact
manifest verifier before any backend/WING WebSocket opens. A non-empty
`failed_checks` list means the no-WING artifact bundle is not clean enough to
support the intended read-only preflight.

## Options considered

1. Keep only the generic
   `prestart_summary_artifact_manifest_verifier_failed` violation.
2. Add a dedicated violation when verifier `failed_checks` is non-empty.

## Decision

Add `prestart_summary_artifact_manifest_verifier_failed_checks_not_empty` when
the verifier reports one or more failed checks.

## Why this won

The failed-check list is the most direct operator repair target. Keeping it as a
named violation makes pre-WebSocket diagnostics more actionable without
weakening the existing fail-closed behavior.

## Rejected alternatives

Relying on generic verifier failure was rejected because it forces operators to
infer the root cause from detail fields instead of seeing the failed-check
condition in the violation list.

## Implementation plan

- Extend `tools/wing_readonly_preflight_runner.py` with the dedicated
  failed-checks violation.
- Add a pre-WebSocket regression test that mocks a non-empty `failed_checks`
  list.

## Test plan

- Focused runner verifier tests.
- Focused WING/no-WING tests.
- Standard full pytest command.

## Risks and rollback

Risk is limited to stricter offline/pre-WebSocket diagnostics. Rollback is to
remove the dedicated failed-checks violation while preserving the generic
verifier failure.
