# ADR: ML-2026-10-08-1500 search reconciliation

- status: accepted for research queue
- date: `2026-10-08`
- decision owner: Mixing Learning research process
- auto_apply: `false`

## Context

The 15:00 search found four previously unindexed rock-drum video IDs, two fully readable official companion articles, and one Cambridge multitrack catalogue entry. No permitted transcript was available, and all scientific hits were existing source families.

## Decision

1. Store the four videos as queued metadata/companion-text cards, never as transcript-complete reviews.
2. Store the two companion articles as vendor-educational Source Cards, not independent scientific confirmation.
3. Accept only the route-measurement workflow as a bounded candidate rule. Reject the article's 15-sample/96-kHz example and demonstrated compressor controls as transferable defaults.
4. Add a one-factor, loudness-matched, `not_run` A/B plan for residual parallel-path delay correction, conditional on an actual defined route.
5. Store Fytakyte's *High Anxiety* as catalogue-only; recheck the entry terms before obtaining or using raw tracks.
6. Keep all rules `auto_apply:false`; make no audio, DSP, model or runtime changes.

## Consequences

The next deep review can focus on permitted transcripts for the two highest-value videos and on the Cambridge entry's rights. The project gains a reproducible alignment test without promoting device/session numbers into production rules. The highest-priority executable experiment remains `EXP-ML-20261001-1800-01`.
