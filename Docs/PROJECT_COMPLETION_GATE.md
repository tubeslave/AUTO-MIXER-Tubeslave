# Project Completion Gate

Status: authoritative completion gate

Automixer может перейти в `COMPLETE` только при evidence-backed выполнении всех
условий ниже. Если требуется hardware, listening или owner validation и ее нет,
статус остается `WAITING_FOR_EXTERNAL_VALIDATION`, а не `COMPLETE`.

## Gate Conditions

1. Все mandatory capabilities из `Docs/CAPABILITY_MATRIX.md` достигли approved
   target maturity.
2. Analysis Mode, Assisted Mixing Mode, Supervised AutoMix Mode и Full
   Autonomous Mixing Mode реализованы, интегрированы и проверены согласно их
   safety class.
3. WING Rack read/write protocol coverage подтвержден валидированными
   protocol paths и staged real-hardware evidence.
4. Ни одна model, LLM, classifier или policy не имеет прямого bypass вокруг
   deterministic Safety Kernel.
5. End-to-end soundcheck, studio и concert scenarios проходят offline, replay,
   simulator, fault-injection и integration evaluations.
6. High-risk console actions прошли staged real-hardware testing под явным
   human supervision.
7. Quality evaluation meets approved thresholds по:
   - objective audio metrics;
   - reference-mix comparison;
   - repeatability and stability;
   - blind listening tests;
   - expert assessment;
   - clipping and overload statistics;
   - masking and intelligibility;
   - control latency;
   - parameter-oscillation rate;
   - recovery and failure behavior.
8. Нет unresolved critical safety defect.
9. Recovery, rollback, emergency-stop and audit behavior verified.
10. Release, operation, calibration and recovery documentation exists.

## Threshold Policy

Текущие quality thresholds не утверждены. Existing baselines можно измерять и
предлагать candidate thresholds, но они имеют статус `PENDING_APPROVAL`, пока
product owner явно не утвердит их.

Candidate threshold policies must be evidence-bound and owner-approved before
they can be used for replay, integration or completion-gate promotion. Complete
candidate threshold evidence alone cannot grant live authority, close readiness
or replace listening, expert, hardware or owner validation.

Replay manifests must consume the quality threshold policy contract result
instead of loose threshold status fields before claiming replay-promotion
readiness.

## Non-completion states

- `RUNNING`: есть dependency-ready autonomous work.
- `READY_FOR_NEXT`: milestone evidence собран, нужен independent verification.
- `WAITING_FOR_EXTERNAL_VALIDATION`: autonomous work для данной ветки исчерпан,
  нужен human/hardware/listening decision.
- `BLOCKED`: один и тот же blocker подтвержден после трех разных попыток.
- `STOPPED`: обнаружен `ops/autopilot/STOP`.

`COMPLETE` запрещен, если любая строка mandatory capability находится ниже
approved target maturity или имеет `UNKNOWN` в required evidence.
