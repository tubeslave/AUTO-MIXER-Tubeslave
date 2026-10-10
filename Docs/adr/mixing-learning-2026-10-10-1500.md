# ADR: Mixing Learning multitrack validation gate — 2026-10-10 15:00 MSK

## 1. Context

The project now has a structured evidence pack for the project-owned 48-track song «Пена». It records 21 one-factor A/B pairs with raw and loudness-matched files, hashes and QC measurements, but no Dmitry listening verdict. The user also made the project-wide requirement explicit: incoming information must be checked on a multitrack.

## 2. Options considered

1. Accept literature and engineer demonstrations directly as project rules.
2. Treat objective render/QC as sufficient validation.
3. Require a project multitrack, one-factor A/B, loudness matching, protected-quality checks and a recorded human verdict before acceptance; require another independent multitrack before generalization.

## 3. Decision

Choose option 3. Sources may create candidate rules and experiments, but they remain `auto_apply:false`. A rendered/measured pair is `pending_listening`, not accepted. One-song preference is local evidence only.

## 4. Why this won

It preserves causal attribution, prevents loudness bias, uses project-owned audio, separates file correctness from musical preference, and keeps style-dependent choices under Dmitry's control.

## 5. Rejected alternatives

- Direct promotion from papers or videos: source evidence does not establish transfer to this production context.
- Objective-only validation: loudness, peaks, clipping and spectral metrics cannot decide arrangement role, groove, tone or preference.
- Multi-factor shootouts: they obscure which change caused the result.
- Generalization from «Пена» alone: one composition cannot establish a universal rule.

## 6. Implementation plan

- Record the policy as `KC-ML-20261010-1500-01`.
- Start with `PENA-AB-01`, the parallel-drum-density pair adapted from `EXP-ML-20261001-1800-01`.
- Run three randomized loudness-matched passes and record accept/reject/ambiguous.
- Process the other 20 pairs one at a time to limit fatigue.
- Repeat accepted candidates on a second independent project-owned/licensed multitrack before broad promotion.

## 7. Test plan

For each candidate retain raw actual-level renders and separate BS.1770-matched X/Y copies. Verify hashes, decode, finite samples, clipping, integrated/section loudness, true peak and technique-specific metrics. Predeclare protected qualities and cancellation criteria. Human acceptance requires consistent preference after matching and no protected-quality regression.

## 8. Risks and rollback

Risks include listener fatigue, blind-map leakage, overfitting to one arrangement and treating the new local baseline as already approved. Rollback is to keep the candidate queued, record an ambiguous or rejected verdict, and make no DSP/runtime change. No automatic promotion is permitted.
