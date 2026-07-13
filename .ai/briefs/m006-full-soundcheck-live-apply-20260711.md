# M006-FULL: полный supervised live apply для саундчека

## Запрос владельца

Во время реального саундчека Automixer должен не только анализировать, но и
регулировать все нужные модули и отправлять подтверждённые коррекции на WING.
Текущий узкий M006, допускающий только `fader` и `gain-down`, не соответствует
этой конечной цели и не должен подписываться как окончательное решение.

## Подтверждённая исходная точка

- M005 physical cable-loss/stale-state proof завершён на реальном WING:
  `artifacts/wing_timeout_disconnect_proof/20260711T131505Z_wing_m005_disconnect_timeout_stale_proof/summary.json`.
- CH3 до и после proof: `-12.099998474121094 dB`; write не применился,
  rollback не понадобился.
- Приложение после proof возвращено в `assist`, WING reconnect выполнен в
  `observation_only=true`.
- Канонический `apply_operator_proposal` и `_apply_manual_console_write`
  транзакционно обслуживают только простые manual kinds.
- `GatedCorrectionActionExecutor` уже моделирует широкий набор typed actions,
  но не является допустимым вторым live writer.
- DSP/transport-методы для EQ, dynamics, gate, phase/delay, FX, buses, DCA,
  matrix, routing, scenes и master частично существуют, но большинство live
  семейств не имеют принятого end-to-end hardware proof.

## Требуемый результат

1. Один Product Layer transaction kernel для всех soundcheck-коррекций.
2. Строгий registry семейств и параметров: mapping, bounds, dependencies,
   pre-read, physical readback, rollback, emergency-stop и audio recheck.
3. Soundcheck формирует абсолютные, идемпотентные proposals и выполняет их
   последовательно после явного operator approval.
4. Поддерживаемые семейства:
   - gain/trim, fader;
   - HPF/EQ;
   - gate/expander;
   - compressor/dynamics/de-esser;
   - mute, pan, polarity, delay;
   - FX models/parameters, inserts, sends/returns;
   - bus/group/DCA/matrix;
   - routing and scenes/presets;
   - master/output processing.
5. Каждая семья остаётся `proposal_only` или `proof_candidate`, пока её
   собственный real-WING apply/readback/rollback/emergency/audio proof не принят.
6. Никаких прямых обходов через AutoSoundcheck, voice, legacy tools или raw OSC.
7. Frontend показывает реальные family status, blockers, transaction/readback,
   rollback и audio-recheck результат.

## Safety invariants

- Один writer lease и один live write entrypoint: `apply_operator_proposal`.
- Все новые live actions только supervised; Full Auto остаётся закрытым.
- Fader не выше 0 dBFS без отдельного operator intent.
- Gain-up запрещён по короткому/bleed-доминантному материалу и требует true-peak evidence.
- EQ first proof cut-only; boosts требуют отдельного headroom proof.
- Dynamics first proof makeup-neutral; enable/model/mix меняются только при model readback.
- Gate first proof bounded threshold; chatter/false-open audio recheck обязателен.
- Mute/routing/scenes/master считаются high-risk и требуют отдельного подтверждения перед каждой транзакцией.
- Любой timeout, stale/cache-only readback, mismatch или audio regression блокирует последующие шаги и запускает rollback/emergency policy.
- Real hardware testing выполняется только при физическом присутствии оператора.

## Non-goals первого checkpoint

- Не объявлять все семейства hardware-proven до реальных proof.
- Не включать unattended/full-auto console control.
- Не удалять существующие generated artifacts и отчёты.
- Не добавлять production dependencies.

## Required evidence

- Unit and integration tests per family contract.
- Replay/fault tests for disconnect, stale state, partial apply and rollback.
- no-WING gate proving no authority is widened from offline evidence.
- Real WING staged artifact per family with exact address/value binding.
- Master-loop audio A/B recheck where the correction is audible in Main LR.
- Operator-visible report and final readiness matrix update from accepted artifacts only.

## Dirty worktree boundary

The task depends on uncommitted M005 fixes in:

- `backend/handlers/mixer_handlers.py`
- `backend/server.py`
- `backend/wing_client.py`
- `tools/wing_timeout_disconnect_proof_runner.py`
- corresponding tests

Existing untracked `artifacts/*` and `reports/*` belong to prior evidence and
must not be deleted, rewritten or staged as broad implementation output.
