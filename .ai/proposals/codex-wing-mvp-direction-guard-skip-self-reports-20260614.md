# Proposal: Skip generated direction guard reports in evidence scoring

## Short thesis

Direction guard reports are diagnostic outputs, not new work evidence, so the
guard should skip them when scoring the next cycle.

## What I understood

The guard scans recent report files. A generated guard report can include prior
summaries and policy vocabulary, so rescanning it can create a false mixed
signal even when the actual work is still on the WING read-only blocker.

## Proposed solution

- Skip generated direction guard reports inside `analyze_evidence`.
- Add regression coverage for generated Markdown and JSON guard reports.
- Keep real WING proof evidence scoring unchanged.
- Register the change as WING MVP rail evidence.

## Likely files to touch

- `tools/codex_autoloop/codex_direction_guard.py`
- `tests/test_codex_direction_guard.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Filtering only specific sections was rejected because JSON snippets can be
truncated. Changing score thresholds was rejected because it would hide the
underlying evidence-source problem.

## Risks

Skipping generated guard reports lowers evidence counts in tests that relied on
derivative target evidence. Tests now use real WING target items when expecting
`on_target`.

## Test plan

- Focused direction guard tests.
- Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may prefer to parse every generated report more carefully. I recommend
skipping these files because they are derived diagnostics, not primary work.
