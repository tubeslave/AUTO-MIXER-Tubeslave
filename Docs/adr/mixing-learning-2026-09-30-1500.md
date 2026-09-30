# ADR: add PAQM-loss evidence as evaluation guidance only

- **Status:** Accepted
- **Date:** 2026-09-30
- **Decision ID:** `mixing-learning-2026-09-30-1500`
- **Research update:** `ML-2026-09-30-1500`

## Context

A missing June 2026 JAES source provides full open author-manuscript evidence for a differentiable PAQM training loss, but its task is bandwidth/codec restoration, its benefit changes with material and degradation, and its repository documents discarded invalid batches. Three useful rock-vocal/general-mix videos have metadata but no permitted transcript.

## Options considered

1. Promote PAQM or source training settings into active mixing rules.
2. Add only the bibliographic record and leave implications unstructured.
3. Capture the source as evaluation guidance, preserve task boundaries, queue untranscribed videos and plan a local controlled metric sanity check.

## Decision

Choose option 3. Publish one source card, one artifact audit, five Knowledge Cards, three candidate rules, three metadata-only video cards and one `not_run` experiment. Treat DOI and arXiv manuscript as one version family. Keep all rules `auto_apply:false` and do not modify runtime, DSP, models or audio.

## Why this won

It preserves useful method and listening evidence while preventing metric circularity, restoration-to-mixing transfer, undocumented batch selection and transcript-free technique claims. It also keeps the next practical drum-room A/B ahead of the auxiliary metric test.

## Rejected alternatives

- Runtime adoption was rejected because no mixing task was tested, artifact rights/revision are incomplete and invalid-batch behavior is unresolved.
- Bibliography-only capture was rejected because the source contains concrete evaluation and reproducibility limits that materially constrain future experiments.
- Treating metadata-only videos as studied was rejected because captions/transcripts were unavailable.

## Implementation plan

Add the dated Markdown report and JSON patch, append the shared index row, record stable source/video IDs, and preserve the prior queue plus three new queued videos.

## Test plan

- Parse the JSON patch.
- Verify unique IDs, date/slot fields, queue arithmetic and predecessor links.
- Review the PR diff to confirm only research documentation/index files changed.
- Require repository CI before merge.

## Risks and rollback

Risk: source-specific restoration results could be read as production presets. Mitigation: settings are provenance-only, direct mixing transfer confidence is low and rules remain inactive. Risk: repository availability may be mistaken for reusable licensing. Mitigation: licence remains unverified and execution is blocked. Rollback: revert the documentation commit; no audio or runtime state requires restoration.
