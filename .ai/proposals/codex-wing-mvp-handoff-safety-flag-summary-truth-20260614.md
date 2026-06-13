# Proposal: Preserve unsafe handoff safety flags in summaries

## Short thesis

The no-WING handoff summary should report actual packet safety flags, including
unsafe `true` values, so failed artifacts remain auditable.

## What I understood

The WING read-only handoff remains an offline/no-WING gate. Its summary is used
to decide whether the next manual hardware step is even eligible, so it must not
hide a packet that claims to write, mutate console state, query the console, or
execute tests.

## Proposed solution

- Change the handoff summary exporter to derive `live_write`,
  `mutates_console_state`, `queries_console`, and `executes_tests` from
  validation detail instead of hardcoded `false`.
- Extend the unsafe handoff packet test so it sets those fields to `true` and
  verifies the JSON and Markdown summaries preserve them.
- Add the report to the WING MVP blocker evidence list and rail docs.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Leaving the fields hardcoded to `false` was rejected because it can make the
operator-facing summary safer than the failed raw packet. Adding separate raw
fields was deferred because the existing fields are what operators and rail docs
already inspect.

## Risks

Failed unsafe summaries may now show `true` values that are intentionally
alarming. That is desired audit behavior and does not grant live authority.

## Test plan

- Focused unsafe handoff summary test.
- Focused no-WING/Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may prefer a broader schema split between normalized gate state and raw
packet state. I recommend preserving the existing field names first because
those are the fields already displayed in the current summary.
