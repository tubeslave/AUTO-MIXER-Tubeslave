# ADR: Mixing Learning search update 2026-10-03 15:00 MSK

- **Status:** accepted research-base update
- **Update:** `ML-2026-10-03-1500`
- **Decision scope:** source queue, video cards, bounded hypotheses and unrun experiment only

## Decision

Add two DAFx-26 sources to the deep-review queue at their exact limited reading depth; do not convert their abstract claims into production rules. Map DAFx-26 paper 30 to the already reviewed `arXiv:2606.18573` lineage rather than counting it as independent evidence.

Upgrade `youtube:IfmB4UAg18g` to transcript-complete and retain its observations as first-party, single-case production evidence. Add `youtube:s4IGWJ1J3WM` and `youtube:At25V53hDYQ` only as metadata-level queued sources because their visible transcript panels did not yield readable segments.

Adopt two candidate test policies—unchanged demo tone as the comparator for reamping, and a mono-gated center-clear width test for a non-foundational guitar texture—with `auto_apply:false`. Add `EXP-ML-20261003-1500-01` as `not_run`; do not alter audio, DSP, repositories or model behavior.

## Rationale

The source and video evidence is useful for defining controlled tests, but it is heterogeneous and not independently replicated. First-party artist walkthroughs describe successful local decisions, not general production laws. Abstract-only conference discoveries establish relevance, not validity. A conservative queue update therefore preserves traceability without overstating reading depth or evidence strength.

## Consequences

- The video queue becomes 84 sources: 7 transcript-complete, 0 partial and 77 metadata-only.
- The next deep review should prioritize `DAFx26_paper_04` and `DAFx26_paper_01`.
- The next immediately executable project experiment remains `EXP-ML-20261001-1800-01`; no new test displaces it.
- All new rules remain `candidate`, `auto_apply:false`; all experiments remain `not_run`.
