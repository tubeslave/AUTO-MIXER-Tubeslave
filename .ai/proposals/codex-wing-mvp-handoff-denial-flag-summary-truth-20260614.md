# Proposal: Preserve unsafe handoff denial flags in summaries

## Short Thesis

The no-WING handoff summary should report actual packet flag values, including
unsafe `true` values, so failed artifacts remain auditable.

## What I Understood

The WING read-only handoff is still offline/no-WING evidence. It must fail
closed before a real WING session and should never visually hide an unsafe
packet state from the operator summary.

## Proposed Solution

- Change the handoff summary exporter to derive denial/live-authority booleans
  from validation detail instead of hardcoded `false`.
- Extend the unsafe handoff packet test so it sets unsafe `true` values and
  verifies they remain visible in JSON and Markdown summaries.
- Add the report to the WING MVP blocker evidence list.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives Considered

Leaving only the validation failure was rejected because the operator-facing
summary can be read independently from raw violations. Adding a separate raw
flag section was deferred because preserving the existing fields is enough for
this slice.

## Risks

Older unsafe test fixtures may now show `true` in summaries. That is intended
and makes failed packets more transparent.

## Test Plan

- Focused unsafe handoff summary test.
- Focused no-WING/Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Where The Other Agent May Disagree

Kimi may prefer adding parallel raw fields. I recommend preserving the existing
field names first because those are the fields already used in the runbook and
blocker table.
