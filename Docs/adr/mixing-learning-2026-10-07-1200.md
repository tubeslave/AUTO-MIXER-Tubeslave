# ADR — Mixing Learning deep review 2026-10-07 12:00 MSK

## Context

The 09:00 search added a DAFx-2014 method for offline threshold/ratio estimation. Its compressor topology and automation lineage depend on Giannoulis, Massberg and Reiss (JAES 2013), which was not represented by a source card. The official AES record also announces a February-2014 correction to Eq. 3.

## Options considered

1. Treat the DAFx paper as self-contained and leave the dependency unreviewed.
2. Add the JAES paper as independent confirmation of mastering preference.
3. Read the full paper, map the dependency, record its study boundary and block equation-parity claims until the official correction or verified reference code is available.

## Decision

Choose option 3. Add `SRC-ML-20261007-1200-01`, three Knowledge Cards and one bounded candidate rule; revise the 09:00 rule and experiment; add a controller-method A/B that stays `not_run_blocked`.

## Why this won

The paper is valuable implementation evidence but its 16-participant method-of-adjustment study on four isolated tracks is not an independent loudness-matched full-chain preference test. Recording the correction gap prevents a paper-title citation from being mistaken for verified code parity.

## Rejected alternatives

- Leaving the dependency unreviewed would hide the origin and limits of the adaptive controller.
- Counting it as mastering-preference replication would inflate evidence and ignore different endpoints.
- Copying Eq. 3 from the accessible PDF despite the official correction notice would create an avoidable reproducibility risk.

## Implementation plan

- Publish the Markdown report and JSON patch under `Docs/mixing_learning_updates/`.
- Add this ADR under `Docs/adr/`.
- Append the update to the shared index.
- Leave DSP, model, audio, runtime registries and production code unchanged.

## Test plan

- Parse the JSON patch.
- Verify report/JSON/ADR and index links after merge.
- Verify every rule has `auto_apply:false` and both experiments are not run.
- Verify the source is not duplicated by title/URL and the 2017 DOI comparator is only referenced.

## Risks and rollback

Risk: the correction notice may be misread as proof that the paper's overall conclusions are invalid. It only blocks exact equation-parity claims until the corrected content is verified. Rollback is removal of the documentation artifacts; production behavior is unaffected.
