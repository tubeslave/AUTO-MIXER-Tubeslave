# ADR: ML-2026-10-16-0900

## Context

Search rotation covered plugin-render provenance, small-difference listening design, bass/distorted-bass practice and new video candidates. The prior queue already contained executable listening and PDC-gated snare work, so this update must not displace it with an untested style recipe.

## Decision

1. Add `vst-bench-ml` as a full-text provenance source and public-artifact audit, but do not treat hash repeatability as portability, scientific validity or perceptual quality.
2. Add the Opus ABC/HR paper as listening-test design evidence only. Preserve the distinction between difference and equivalence; keep its bitrate result outside project delivery rules.
3. Add the URM bass article and Nolly companion as engineer-specific educational sources. Reject promotional absolutes and keep their frequency ranges as source-origin search bounds, not defaults.
4. Queue `youtube:00kI0Ashqp8` and `youtube:Oq62X1hQoFY`; do not mark them studied without a permitted transcript or other content access.
5. Plan one-factor distorted-bass return A/B after existing executable priorities. No DSP or repository implementation change follows from this search.

## Consequences

- Plugin-rendered future evidence gains a stronger identity/manifest/failure gate.
- Listening claims cannot use non-significance as proof of equivalence.
- Bass distortion remains a shadow candidate requiring latency/polarity, loudness and protected-quality checks.
- All rules remain `auto_apply:false`; `EXP-ML-20261016-0900-01` remains `not_run`.

