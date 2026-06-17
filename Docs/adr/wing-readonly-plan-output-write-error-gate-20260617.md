# ADR: WING read-only plan-only blocks on plan-output write errors

## Context

`--plan-only --plan-output` is the operator-reviewed expected-plan step before
any real WING read-only WebSocket preflight. If the runner cannot write the
requested plan output file, the expected-plan evidence chain is incomplete.

## Options considered

1. Let plan output write errors raise out of the runner.
2. Print a structured blocked plan-only JSON when the output file cannot be
   written.

## Decision

Use option 2. Plan-only output write failures now produce
`reason: plan_output_write_error`, `plan_output_write_error`, and
`plan_output_violations` while keeping execution and live authority denied.

## Why this won

The operator needs a structured stop reason even when the requested artifact
cannot be written. A traceback does not provide a stable review surface and can
hide the fact that no backend/WING WebSocket was opened.

## Rejected alternatives

Treating stdout-only output as success was rejected because the requested
artifact is part of the preflight evidence chain.

## Implementation plan

Make the shared plan-output writer return a structured error. Convert that
error into a blocked plan-only output before any WebSocket can be opened.

## Test plan

Add a regression where `--plan-output` points at a directory. Run the targeted
test, focused WING/no-WING suite, no-WING quick, `git diff --check`, and the
standard pytest gate.

## Risks and rollback

Risk is low because this only changes plan-only behavior and does not contact
the backend or WING. Rollback is to let output write exceptions propagate again.
