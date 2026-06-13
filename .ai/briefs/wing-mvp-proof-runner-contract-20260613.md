# WING MVP Proof Runner Contract

## Task

Bind the read-only WING preflight proof runner id to a shared contract constant.

## Context

The WING read-only preflight artifact already shares its schema id through
`backend/wing_readonly_preflight_contract.py`, but the proof runner id is still
duplicated in runner, evidence intake, and no-WING plan validation.

## Safety Boundary

This is offline/no-WING contract hardening. It does not contact hardware,
start a backend listener, grant live authority, or close readiness.

## Expected Outcome

- The accepted proof runner id has one contract source.
- Runner, evidence intake, and no-WING plan validation use that source.
- no-WING release checks fail closed if runner/evidence proof-runner identity
  drifts from the contract.
