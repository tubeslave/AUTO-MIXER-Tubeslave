# ADR: Mixing Learning search 2026-10-05 15:00 MSK

- decision ID: ML-2026-10-05-1500
- status: accepted as knowledge-base update; no automatic production action
- mode: search
- preceding update: ML-2026-10-05-1200
- preceding successful search: ML-2026-10-04-1500

## Decision

Retain one full-relevant-text drum-synthesis preprint as evidence for evaluation design, queue two abstract-only scientific sources for deep review, and add two metadata-only videos without claiming their content was studied.

Adopt one bounded candidate rule: drum replacement/rendering should preserve event-level timing, velocity and role, and any approved timing move should retain the multi-mic event as a group. This is not an editing preset and is not auto-applied.

Do not promote the paper's ±50 ms onset tolerance or the abstract-only P-center paper's reported 40 ms condition into production thresholds. Both values belong to their experimental protocols.

## Evidence boundary

The full-read preprint uses aligned audio/MIDI from E-GMD and compares neural audio codecs under objective metrics. It contains no human listening study or formal statistical test and does not evaluate real rock multitracks, bleed, microphone phase alignment or full-mix groove preference.

The P-center paper and Separate-and-Detect work were not available at full-text depth in this run. Their abstract-level claims remain queued, with citation_check:partial.

Video transcripts were unavailable. No settings, techniques or timestamps were inferred from titles or descriptions.

## Consequences

- RULE-ML-20261005-1500-01 remains auto_apply:false.
- EXP-ML-20261005-1500-01 remains not_run.
- The next executable project test is still EXP-ML-20261001-1800-01.
- The next deep-review priority is DOI 10.1111/nyas.70306, followed by arxiv:2608.01093 and the public drum-grid artifact.
- No audio, DSP, runtime, repository code or model state changes.
