# ADR: WING real artifact expected-artifacts binding

## Context

The WING read-only runner writes an offline plan containing the expected
`summary.json` and `events.jsonl` artifact globs. The evidence review already
checks hashes, read-only event order, request IDs, physical readbacks, and
expected-plan metadata. It did not yet prove that the reviewed real artifact
paths were the paths declared in `expected_artifacts`.

## Options Considered

1. Leave artifact path confirmation to the manual checklist.
2. Render expected artifacts only in the review report.
3. Fail closed when the expected plan and actual artifact paths do not match.

## Decision

Fail closed when file-backed WING read-only candidates do not bind to the
expected-plan `expected_artifacts` after `<run_id>` expansion.

## Why This Won

The artifact path is part of the proof identity. If the operator approved one
artifact root but the review accepts files from another path, the candidate may
not represent the intended WING run. A fail-closed evidence check preserves the
manual review gate without adding any live authority.

## Rejected Alternatives

Manual-only confirmation was rejected because the code can cheaply catch stale
or mismatched paths before review. Report-only rendering was rejected because it
would still allow a mismatched candidate to appear clean.

## Implementation Plan

- Validate expected-plan `artifact_root` and `expected_artifacts`.
- Compare expanded expected summary/event paths to the artifact's recorded
  `summary_path` and `events_path`.
- Surface expected artifact paths and match flags in evidence and review output.
- Add regressions for stale expected artifact globs.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py`
- Adjacent WING/no-WING tests.
- Full test suite before merge.

## Risks And Rollback

The stricter path binding can reject old non-runner-shaped fixtures. Rollback is
to remove the path-binding check while keeping the report-only fields.
