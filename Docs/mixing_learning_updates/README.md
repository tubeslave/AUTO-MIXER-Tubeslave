# Mixing Learning: shared research index

This directory is part of the shared knowledge base on the repository's default branch, `master`. Publication here makes research available for agents to read; it does not prove that a running agent or another chat has already read it.

## Updates

| Update | Materials | State |
| --- | --- | --- |
| 2026-09-10 15:00 MSK | [Research and video queue](2026-09-10_1500.md) | Existing research update; preserved unchanged |
| ML-SPATIAL-2026-09-11-155446 | [Report](ML-SPATIAL-2026-09-11-155446.md), [structured package](ML-SPATIAL-2026-09-11-155446.json), [publication manifest and hashes](ML-SPATIAL-2026-09-11-155446.publication.json) | Shared research package; 11 source cards, 8 Knowledge Cards, 8 candidate rules, 3 queued videos, 4 unrun experiments |

## Spatial package: reading and safety

Read the publication manifest first. The report and JSON are byte-preserved originals from the research run. Their statements about unavailable writers and `persistence=false` describe the earlier export, not whether the package is now present in this repository.

The JSON arrays are the package's source/card/rule/experiment registries. Stable IDs and JSON pointers are recorded in the publication manifest. Do not treat publication as audio validation: all proposed rules remain `candidate`, `auto_apply:false`, and all audio experiments remain `not_run`. No mandatory vocal priority or automatic effect settings are introduced.

This transfer intentionally does not inject raw candidates into `backend/source_knowledge/data/rules.jsonl`: the package uses categorical confidence and a richer research schema, whereas `SourceRule` expects numeric confidence and action templates. A later reviewed adapter must preserve the original confidence labels, perform full cross-registry DOI/URL/video-ID and rule deduplication, and register source IDs consistently before runtime retrieval is enabled. Existing source registries, runtime configuration, DSP, OSC, audio and feedback remain unchanged.

Earlier `R-20260911-1200-*` and `R-20260911-1500-*` references in the package refer to research updates seen in the original study. Their presence in the runtime registry is not asserted by this transfer.

## Handoff for mixing agents

ML-SPATIAL-2026-09-11-155446: distinguish dry position, source width, relative depth and ambience. Pre-delay is not a distance scale. Choose pan/width operators and musical priority by section role. Preserve mono compatibility, attack, groove and the intended rhythm-guitar foundation. Next proposed test: EXP-SP-20260911-155446-01, wet contribution versus pre-delay, blind and loudness-matched; `not_run`.
