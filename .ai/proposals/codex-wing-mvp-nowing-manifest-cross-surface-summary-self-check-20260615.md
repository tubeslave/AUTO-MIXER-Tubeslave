# Proposal: no-WING Manifest Cross-Surface Summary Self-Check

## Short thesis

Make no-WING `summary.json` and `summary.md` compute manifest contract
cross-surface verdicts instead of declaring success by construction.

## What I understood

The first WING MVP blocker still needs real read-only WING evidence. Offline
no-WING artifacts remain useful only if they fail closed when their own
operator-facing surfaces drift.

## Proposed solution

- Factor manifest contract cross-surface checking into an in-memory helper that
  can inspect a summary dictionary and Markdown text before files are written.
- Apply that helper in `write_summary`.
- Render the computed cross-surface SHA match and mismatch fields back into
  `summary.md`.
- Add a regression test for a stale manifest contract digest.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`

## Alternatives considered

- Rely only on detached `artifact-manifest.json`: rejected because the primary
  no-WING summary surfaces could still overstate success.
- Write the manifest first and feed it back into the summaries: rejected
  because the manifest hashes the summary files, creating avoidable ordering
  complexity.

## Risks

The check only covers the contract fields listed in the manifest contract. New
cross-surface fields still need to be added to the contract list deliberately.

## Test plan

- Targeted stale-digest regression test.
- Focused no-WING, Project Director, and WING MVP rails tests.
- no-WING quick check.
- Standard pytest.

## Where the other agent may disagree

Kimi may prefer a single summary/manifest schema renderer. That would be a good
future cleanup, but the in-memory checker closes the current stale verdict risk
without broad rewrites.
