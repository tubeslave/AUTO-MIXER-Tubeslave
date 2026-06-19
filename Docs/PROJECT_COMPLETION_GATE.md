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

Recovery policy candidates must be evidence-bound and owner-approved before
they can be used for simulator fault-recovery promotion. Complete candidate
recovery policy evidence alone cannot grant live authority, close readiness,
replace emergency-stop or rollback proof, or replace hardware validation.

Fault-recovery manifests must consume the recovery policy contract result
instead of loose recovery_policy status fields before claiming simulator
fault-recovery promotion readiness.

Integration policy candidates must be evidence-bound and owner-approved before
they can be used for offline integration promotion. Complete candidate
integration policy evidence alone cannot grant live authority, close readiness,
replace child replay/fault-recovery evidence, or replace hardware validation.

Integration bundle manifests must consume the integration policy contract result
instead of loose integration_policy status fields before claiming offline
integration promotion readiness.

Safety-kernel trace candidates must be offline, complete and audited before
they can be used for integration or completion-gate promotion. Complete
candidate safety-kernel trace evidence alone cannot grant live authority, close
readiness, bypass deterministic safety checks, or replace hardware validation.

Integration bundle manifests must consume the safety-kernel trace contract
result instead of loose safety_kernel_trace_bound checks before claiming offline
integration promotion readiness.

No-live-transport evidence candidates must be offline, artifact-only and
audited before they can be used for integration or completion-gate promotion.
Complete no-live-transport evidence alone cannot grant live authority, close
readiness, contact WING/backend, send live writes, or replace hardware
validation.

Integration bundle manifests must consume the no-live-transport contract
result instead of loose no_live_transport checks before claiming offline
integration promotion readiness.

Operator report candidates must be offline, evidence-bound and
operator-visible before they can be used for integration or completion-gate
promotion. Complete operator-report evidence alone cannot grant live authority,
close readiness, contact WING/backend, replace accepted real proof, or replace
operator/hardware validation.

Integration bundle manifests must consume the operator-report contract result
instead of loose operator_report_bound checks before claiming offline
integration promotion readiness.

Autonomy promotion readiness snapshots must consume the offline integration
bundle contract result before treating offline integration evidence as
promotion-planning input. Passing offline integration evidence alone cannot
grant live authority, close readiness, mark Full Autonomous ready, or replace
real hardware, listening, expert, owner, rollback or emergency-stop proof.

Operator-facing module status adapters must surface offline integration bundle
readiness, blockers and authority-denial status as read-only evidence only.
Frontend-visible integration bundle readiness cannot grant live authority,
close readiness, contact WING/backend, mark Full Autonomous ready, or replace
accepted real hardware/operator validation.

Main pipeline preview summaries must surface offline integration bundle
readiness, blockers and authority-denial status as read-only evidence only.
Preview-visible integration bundle readiness cannot queue proposals, grant live
authority, close readiness, contact WING/backend, mark Full Autonomous ready,
or replace accepted real hardware/operator validation.

Director runtime guards must surface offline integration bundle readiness,
blockers and authority-denial status as read-only evidence only. Director-visible
integration bundle readiness cannot queue proposals, grant live authority, close
readiness, contact WING/backend, mark Full Autonomous ready, or replace
accepted real hardware/operator validation.

Director event rows must render offline integration bundle blocker status as
read-only operator explanation only. Event-visible integration bundle status
cannot queue proposals, grant live authority, close readiness, contact
WING/backend, mark Full Autonomous ready, or replace accepted real
hardware/operator validation.

Director Runtime Guard banners must render offline integration bundle blocker
status as read-only operator explanation only. Banner-visible integration bundle
status cannot queue proposals, grant live authority, close readiness, contact
WING/backend, mark Full Autonomous ready, or replace accepted real
hardware/operator validation.

Director current-goal text must render offline integration bundle blocker
status as read-only operator explanation only. Goal-visible integration bundle
status cannot queue proposals, grant live authority, close readiness, contact
WING/backend, mark Full Autonomous ready, apply to console, or replace accepted
real hardware/operator validation.

Director proposal queue badges must render offline integration bundle
promotion-planning blockers as read-only operator explanation only.
Badge-visible integration bundle status cannot queue proposals, grant live
authority, close readiness, contact WING/backend, mark Full Autonomous ready,
apply to console, hide review-only evidence rows, or replace accepted real
hardware/operator validation.

Director Agent Suggestions panels must render offline integration bundle
blocker notices as read-only operator explanation only. Notice-visible
integration bundle status cannot queue proposals, grant live authority, close
readiness, contact WING/backend, mark Full Autonomous ready, apply to console,
hide review-only evidence rows, or replace accepted real hardware/operator
validation.

Task Queue panels must render offline integration bundle blocker notices as
read-only operator explanation before proposal actions. Queue-notice-visible
integration bundle status cannot queue proposals, grant live authority, close
readiness, contact WING/backend, mark Full Autonomous ready, apply to console,
hide review-only evidence rows, or replace accepted real hardware/operator
validation.

Supervised Apply modals must render offline integration bundle blocker notices
as read-only operator explanation before supervised apply review.
Modal-notice-visible integration bundle status cannot queue proposals, grant
live authority, close readiness, contact WING/backend, mark Full Autonomous
ready, apply to console, hide review-only evidence rows, bypass apply gates, or
replace accepted real hardware/operator validation.

Supervised Apply blockers must enforce the blocked state in frontend apply
controls. When offline integration bundle blockers are visible, task-row apply
actions and modal confirm controls must stay blocked. Frontend enforcement
cannot grant live authority, close readiness, contact WING/backend, mark Full
Autonomous ready, apply to console, replace backend safety gates, or replace
accepted real hardware/operator validation.

Supervised Apply confirm callbacks must fail closed before dispatch when
offline integration bundle blockers are visible. Callback-visible enforcement
cannot queue proposals, dispatch supervised apply, grant live authority, close
readiness, contact WING/backend, mark Full Autonomous ready, apply to console,
replace backend safety gates, or replace accepted real hardware/operator
validation.

## Non-completion states

- `RUNNING`: есть dependency-ready autonomous work.
- `READY_FOR_NEXT`: milestone evidence собран, нужен independent verification.
- `WAITING_FOR_EXTERNAL_VALIDATION`: autonomous work для данной ветки исчерпан,
  нужен human/hardware/listening decision.
- `BLOCKED`: один и тот же blocker подтвержден после трех разных попыток.
- `STOPPED`: обнаружен `ops/autopilot/STOP`.

`COMPLETE` запрещен, если любая строка mandatory capability находится ниже
approved target maturity или имеет `UNKNOWN` в required evidence.
