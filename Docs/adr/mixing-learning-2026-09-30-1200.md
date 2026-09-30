# ADR: deep-review SAGE and MERT2/SheetSage2 without runtime adoption

- **Status:** Accepted
- **Date:** 2026-09-30
- **Decision ID:** `mixing-learning-2026-09-30-1200`
- **Research update:** `ML-2026-09-30-1200`

## Context

The 09:00 search queued full evaluation of SAGE and targeted MERT2/SheetSage2 review. These sources concern learned audio representations and transcription, not controlled mixing outcomes. The SAGE paper links a repository that is not currently reachable, while the YuE2-family model cards are available under non-commercial terms and require custom remote code.

## Decision

Publish source-card upgrades, artifact audits, three protocol-only candidate rules and two unrun experiment plans. Keep learned/perceptual, waveform, stereo and listener endpoints separate. Restrict MERT2/SheetSage2 to human-reviewed annotation hypotheses because their documented input is 24-kHz mono and their evaluations do not validate mix control. Block SAGE execution until a reachable, licensed, pinned artifact exists.

Do not modify runtime code, DSP, active rules, model weights or audio. Do not download gated weights or execute GPU jobs.

## Consequences

- Scientific evidence is preserved with method, metric and independence limits.
- The SAGE experiment is explicitly blocked rather than silently treated as reproducible.
- YuE2-family artifacts remain candidates for local non-commercial research only after revision, licence, dependency and remote-code review.
- All rules remain `candidate / auto_apply:false`; all experiments remain `not_run`.

## Validation

- JSON patch parses successfully.
- Update and ADR use unique 12:00 filenames.
- Previous search link and stable source IDs are preserved.
- No code, audio, DSP or runtime file is changed.
