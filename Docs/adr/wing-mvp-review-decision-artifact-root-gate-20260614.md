# ADR: WING MVP Review Decision Artifact Root Gate

Date: 2026-06-14
Status: Accepted

## Context

WING read-only preflight captures now include an explicit `artifact_root`.
Evidence intake requires that root and checks it against the offline plan. The
operator review decision is the next manual acceptance boundary, so it must
also preserve the same root before it can accept a candidate.

## Options considered

1. Keep review decisions bound only by summary/event paths and SHA-256 values.
2. Require `artifact_root` in the decision template and validate it against the
   accepted candidate.
3. Infer the root from `summary_path` inside the decision ledger.

## Decision

Use option 2. The review report and pending template now carry the candidate
`artifact_root`, and the decision ledger rejects missing or mismatched manual
roots.

## Why this won

The operator-edited decision should preserve the same artifact identity that the
runtime summary and evidence row already declared. Explicit root comparison
keeps the manual gate auditable and avoids silently accepting stale copied
paths from another artifact tree.

## Rejected alternatives

- Path/hash-only binding: strong for bytes, but weaker for the capture tree
  identity now required by runtime evidence.
- Derived root comparison: useful as a fallback, but it hides a missing manual
  field instead of forcing the operator record to say what it reviewed.

## Implementation plan

- Add `artifact_root` to the shared review-decision required-field contract.
- Carry `artifact_root` through evidence artifact records and review rows.
- Include `artifact_root` in the pending decision template.
- Validate missing and stale manual roots in the decision ledger.
- Keep readiness/live authority flags denied.

## Test plan

- `PYTHONPATH=backend python -m pytest -q
  tests/test_wing_readonly_preflight_review_decision.py
  tests/test_wing_readonly_preflight_review_tool.py`
- `PYTHONPATH=backend python -m pytest -q
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_wing_readonly_preflight_runner.py`
- Full no-WING release-check module.
- Project Director rails, py_compile, memory lint, and `git diff --check`.

## Risks and rollback

Rollback removes the root field from the template, evidence row, ledger
comparison, tests, and report. The change is offline validation only; it does
not contact hardware, mutate console state, grant live authority, or close
readiness.
