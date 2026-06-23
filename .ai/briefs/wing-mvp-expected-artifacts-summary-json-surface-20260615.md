# Brief: WING MVP Expected Artifacts Summary JSON Surface

Date: 2026-06-15

## Task

Expose the WING read-only handoff expected-artifact contract fields as top-level
no-WING `summary.json` fields.

## Constraints

- Keep `WING read-only connection proof` as the first open blocker.
- Do not contact WING, query a console, start a backend/WING session, or widen
  live authority.
- Mirror the existing handoff-packet expected-artifact contract instead of
  changing artifact readiness rules.
- Preserve readiness closure and live authority denial.

## Expected Evidence

- Top-level `summary.json` exposes `expected_artifacts`,
  `expected_artifact_globs`, `plan_expected_artifacts`, and match verdicts.
- Summary and event-log expected globs remain contract-bound.
- Focused and standard tests pass.
