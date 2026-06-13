# Brief: WING real artifact expected-artifacts binding

## Context

The WING read-only runner already emits `expected_artifacts` in its offline
plan. The artifact evidence review accepted file artifacts by checking hashes,
event logs, and plan metadata, but it did not prove that `summary_path` and
`events_path` matched the artifact globs that the operator approved before the
real run.

## Task

Bind accepted WING read-only candidate artifacts to the `expected_artifacts`
declared in the offline plan.

## Acceptance

- Expected plans must contain the exact summary/event artifact globs for their
  `artifact_root`.
- File-backed candidates fail closed when actual `summary_path` or
  `events_path` do not match those globs after `<run_id>` expansion.
- Review reports surface the expected artifact paths and match flags.
- No backend, WING, OSC, or live-write behavior is added.
