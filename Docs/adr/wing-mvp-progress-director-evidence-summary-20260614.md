# ADR: WING MVP Progress Director Evidence Summary

## Context

The first WING MVP blocker remains open until real WING read-only proof is
captured and reviewed. Project Director computes Current First Command baseline
coverage and selected evidence counts, but the generated
`Docs/reports/MVP_PROGRESS_LATEST.md` did not expose those values directly.

## Options considered

1. Keep coverage and evidence counts only in Project Director JSON.
2. Add the values to `MVP_PROGRESS_LATEST.md` and share evidence extraction with
   Project Director.
3. Build a separate progress JSON artifact.

## Decision

Use option 2.

## Why this won

The latest progress report is already the operator-facing status artifact. Adding
the compact coverage and evidence-count fields makes it auditable without
creating another source of truth.

## Rejected alternatives

- Option 1 leaves the visible progress report weaker than Project Director.
- Option 3 can still be useful later, but it is unnecessary for this small
  auditability gap.

## Implementation plan

- Add shared `extract_evidence_files` support to the progress tool.
- Keep a compatibility wrapper in Project Director.
- Render Current First Command baseline coverage fields in the progress report.
- Render selected evidence file count in the progress report.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Add focused rail assertions.
- Add the report to `Docs/INDEX.md` and the first WING blocker evidence trail.

## Test plan

- Focused Project Director and WING rail tests.
- Project Director JSON check.
- Direction guard JSON check.
- no-WING quick check.
- `tools/memory/memory_lint.py`.
- `git diff --check`.
- Full pytest.

## Risks and rollback

Rollback removes the extra progress report fields and returns Project Director
to its local evidence parser. This does not contact WING, mutate console state,
grant live authority, or close the current blocker.
