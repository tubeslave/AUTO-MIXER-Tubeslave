# ADR: WING read-only plan-only writes blocked output artifacts

## Context

The read-only WING runner uses `--plan-only --plan-output` to produce the
expected request plan before any real backend/WING WebSocket run. Successful
plans were written to `--plan-output`, while blocked plan-only outputs were
printed only to stdout.

## Options considered

1. Keep blocked plan-only output on stdout only.
2. Write blocked plan-only output to `--plan-output` when requested.

## Decision

Use option 2. Blocked plan-only outputs now write the same JSON artifact as
stdout when `--plan-output` is provided.

## Why this won

The operator evidence chain should not depend on terminal scrollback. A blocked
plan-only artifact lets review tooling inspect validation or prestart failure
markers from a stable file before any hardware session is attempted.

## Rejected alternatives

Stdout-only blocked output was simpler, but it made failed planning attempts
harder to audit and compare with the later real-run preflight artifact.

## Implementation plan

Add a shared `write_plan_output_if_requested` helper and use it for both blocked
and ready plan-only branches.

## Test plan

Extend the blocked plan-only validation regression to assert that the file is
written and matches stdout. Run the targeted regression, focused WING/no-WING
suite, no-WING quick, `git diff --check`, and the standard pytest gate.

## Risks and rollback

Risk is low because this only writes the requested plan output file and still
does not open a WebSocket. Rollback is to restrict file writes back to ready
plans only.
