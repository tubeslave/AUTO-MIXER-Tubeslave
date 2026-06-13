# WING MVP Selected Blocker Contract

## Task

Bind the selected WING MVP blocker name and status to shared contract constants.

## Context

The read-only WING preflight plan declares the next MVP blocker it supports.
Runner, evidence intake, no-WING validation, and tests currently duplicate the
selected blocker name and status.

## Safety Boundary

This is offline/no-WING contract hardening. It does not contact hardware,
grant live authority, or close readiness.

## Expected Outcome

- The selected WING MVP blocker and status have one contract source.
- Runner, evidence intake, and no-WING plan validation consume that source.
- Tests prove the constants are used by producer and consumer code.
