# Recognize explicit offline overhead side tokens

## Context

Two explicitly named overhead sources were treated as custom/center in a real
eight-track canonical render. Existing overhead spatial analysis was skipped.

## Options considered

Rename inputs; normalize all filename classification; repair overhead-specific
token interpretation in the existing offline module.

## Decision

Use overhead-specific separator-bounded side matching for classification and
pair selection. Preserve existing overhead DSP profiles and live restrictions.
This is a unilateral local correction pending independent review, not a release
or hardware readiness decision.

## Why this won

It repairs the observed signal path without changing unrelated role parsing,
introducing dependencies or granting live authority.

## Rejected alternatives

Renaming hides the failure. General classifier replacement requires a wider
corpus and is unnecessary for explicit L/R metadata.

## Implementation plan

Write failing filename and pair-selection regressions; share a small pure side
parser inside offline_agent_mix; rerun the real canonical pipeline.

## Test plan

Legacy profiles, underscore/dash/space/dot separators, exact token boundaries,
side selection independent of plan ordering/pan, then full real renders and
standard pytest suite. Source-bound CP05 remains separately unverified.

## Risks and rollback

Wrong input labels remain wrong metadata. Audio appearance/phase can change
because the intended analyzers now run: measure outputs and require listening
before approving mix quality. Revert only the scoped parsing patch if necessary;
never overwrite input audio or console state.
