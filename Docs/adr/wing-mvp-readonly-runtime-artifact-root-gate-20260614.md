# ADR: WING MVP read-only Runtime Artifact Root Gate

Date: 2026-06-14
Status: Accepted

## Context

The WING read-only preflight plan declares `artifact_root` and the expected
summary/event artifact globs. Real-run summaries are reviewed as physical proof
candidates before any supervised fader/gain proof. The summary should therefore
carry the same runtime artifact-root value instead of forcing reviewers to infer
it from paths.

## Options considered

1. Continue inferring artifact root from `summary_path`.
2. Record runtime `artifact_root` in the runner summary and compare it against
   the expected plan.
3. Validate only `expected_artifacts` path globs.

## Decision

Use option 2. The real read-only runner records `artifact_root`, and evidence
intake rejects summaries missing that field or whose referenced offline plan
declares a different root.

## Why this won

The expected plan is the operator-reviewed command contract. Recording the
runtime root in the summary makes the captured artifact self-describing and
turns root drift into an explicit fail-closed violation.

## Rejected alternatives

- Inference-only: path shape can be checked, but the runtime argument remains
  implicit.
- Expected-artifacts-only: catches some path drift but does not prove the
  summary exposed the runtime root used by the runner.

## Implementation plan

- Add `artifact_root` to real read-only runner summaries.
- Add `artifact_root` to evidence expected-plan summary field checks.
- Reject missing summary `artifact_root`.
- Add runner and file-level evidence regression tests.
- Update the read-only preflight runbook acceptance and fail-closed criteria.

## Test plan

- `PYTHONPATH=backend python -m pytest
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_evidence.py -q`
- Full no-WING release-check module.
- Project Director rails, memory lint, and `git diff --check`.

## Risks and rollback

Rollback removes the summary field, evidence mismatch check, tests, and runbook
lines. The change is offline/review-only: it does not contact hardware, mutate
console state, grant live authority, or close readiness.
