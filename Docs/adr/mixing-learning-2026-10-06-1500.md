# ADR — bound DeePAQ to domain-specific secondary evaluation

- Date: 2026-10-06
- Status: Accepted
- Update: ML-2026-10-06-1500

## Context

The search found a peer-reviewed ICASSP 2026 metric paper that was absent from Mixing Learning. DeePAQ reports strong aggregate full-reference correlations but is trained with codec distortions and weak ViSQOL/bitrate labels. Its source-separation results are domain dependent, its non-matching-reference form is weak, and no public code/checkpoint/reproduction package was found.

Two new rock/metal bass videos were also found. Neither yielded content that could be fully read: one transcript is unavailable and one transcript panel remained unreadable/loading despite a successful availability signal. Metadata and chapters are insufficient for technique extraction.

## Decision

Add SRC-ML-20261006-1500-01 as studied_full_text_peer_reviewed and preserve its reported numbers with domain and reference-contract boundaries. Do not treat a DeePAQ score as a general mix-quality, preference or acceptance gate.

Revise RULE-ML-20260930-1500-01 instead of creating a duplicate metric rule. Require domain-stratified reporting, matched-versus-nonmatching reference disclosure, PCC-versus-SRCC separation, less-coupled diagnostics and loudness-matched listening.

Revise EXP-ML-20260930-1500-01 so DeePAQ may be added only after a public licensed implementation/checkpoint and exact inference contract are verified. Keep Dmitry's randomized loudness-matched judgment authoritative and keep the experiment not_run.

Add the two videos as queued_source cards without techniques or numerical settings. Record the DeePAQ artifact search as negative rather than claiming reproducibility.

## Consequences

- DeePAQ evidence can inform evaluation design but cannot auto-accept or auto-apply processing.
- Four-second 24 kHz codec evaluation is not evidence for full-song rock mixing, stereo space, headroom or creative intent.
- ViSQOL-derived weak labels are not independent confirmation of ViSQOL.
- VID-ML-20261006-1500-01 and VID-ML-20261006-1500-02 remain queued until their permitted content can be read.
- RULE-ML-20260930-1500-01 remains auto_apply:false.
- EXP-ML-20260930-1500-01 remains not_run.
- The next executable project experiment remains EXP-ML-20261001-1800-01.
- No audio, DSP, runtime, model, repository code or paid-compute change is authorized.
