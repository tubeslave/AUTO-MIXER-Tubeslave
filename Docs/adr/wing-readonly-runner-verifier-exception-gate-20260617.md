# ADR: WING read-only runner fails closed on verifier execution faults

## Context

Before any WING read-only WebSocket preflight, the runner must execute the
detached no-WING artifact manifest verifier against the artifact directory that
contains `--prestart-summary`. If that verifier crashes or returns a malformed
top-level report, the runner must not proceed toward the backend/WING session.

## Options considered

1. Let verifier execution faults raise out of the runner.
2. Convert verifier execution faults and non-object reports into blocked
   preflight summaries.

## Decision

Use option 2. The runner now records verifier execution errors and non-object
reports as prestart violations before opening any backend/WING WebSocket.

## Why this won

The operator needs a reviewable artifact for every preflight stop. A traceback
does not prove that no WebSocket was opened and gives weaker guidance than a
structured `prestart_summary_missing_or_invalid` summary with verifier error
details.

## Rejected alternatives

Allowing the exception to propagate would still avoid live writes, but it would
break the evidence chain and make repeatable operator review harder.

## Implementation plan

Wrap `verify_no_wing_artifact_manifest(...)` in the runner. On exception, add
`prestart_summary_artifact_manifest_verifier_failed` plus a typed verifier
error violation. If the verifier returns a non-object report, add a dedicated
`prestart_summary_artifact_manifest_verifier_report_not_object` violation.

## Test plan

Add regressions proving that verifier exceptions and non-object reports block
before WebSocket and produce summary artifacts. Run targeted regressions,
focused WING/no-WING tests, no-WING quick, `git diff --check`, and the standard
pytest gate.

## Risks and rollback

Risk is low because this only changes verifier fault handling before any
backend/WING connection attempt. Rollback is to remove the catch and
regressions, returning to traceback behavior.
