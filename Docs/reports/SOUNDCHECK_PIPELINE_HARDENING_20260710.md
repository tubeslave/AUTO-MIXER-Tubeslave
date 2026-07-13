# Soundcheck pipeline hardening — 2026-07-10

> Historical snapshot. Its `92/100` predates discovery of the monolithic
> transaction failure and must not be used as current readiness. See
> `Docs/reports/SOUNDCHECK_PLAN_DECOMPOSITION_20260712.md`.

## Итог

Рабочий supervised soundcheck pipeline доведен до состояния **92/100** на
реальном WING Rack. Он подключается к пульту, читает физическое состояние,
захватывает 48-канальный WING USB, обнаруживает и анализирует сигналы и
формирует предложения. Unattended live apply не включен.

## Что реализовано

1. Local WebSocket по умолчанию; remote bind требует токен до регистрации
   клиента.
2. Canonical AutoSoundcheck и MixingAgent proposal-only на реальном пульте;
   `force_auto_apply` больше не является полномочием.
3. Один system emergency-stop с остановкой engine/agent/controllers и
   generation check перед каждым write/retry.
4. `commanded_state` отделен от физического WING `state`; отправка UDP больше
   не создает ложный readback.
5. OSC throttle доставляет последнее значение после безопасной задержки вместо
   silent drop.
6. Manual write получает `applied` только после нового physical readback;
   timeout/mismatch дает `sent_unverified` и аварийный latch.
7. Rollback связан с UUID транзакции и CAS-проверкой физического текущего
   значения.
8. Missing live true-peak и missing/stale master true-peak readiness блокируются.
9. Observe-only soundcheck не перезаписывает tracked learning profile.
10. CI дополнен frontend tests, production build и Playwright E2E.

## Проверки

- Backend: `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q` —
  `5848 passed, 1 skipped`.
- После последнего observe-only profile fix: 516 профильных/release тестов —
  passed.
- Frontend: `npm test` — 98 passed.
- Frontend production build — passed.
- Playwright Chromium — 9 passed.
- `git diff --check` — passed.
- no-WING prestart и detached manifest verifier — passed.

## Реальный WING/audio evidence

- Console: WING Rack `192.168.1.102:2223`, firmware `3.0.5-4-ga4ea833d`.
- Backend: `connection_mode=wing`, `observation_only=true`.
- Read-only preflight:
  `/tmp/automixer_soundcheck_wing_readonly_20260710/20260709T231521Z_wing_readonly_preflight/summary.json`.
- CH3 physical readback:
  - fader: `-8.9752 dB`;
  - trim/gain: `0.0 dB`;
  - pan: `0.0`;
  - all three: `verified_from_physical_readback`.
- WING USB preview: 48 configured channels, 43 channels with signal meters.
- Canonical AutoSoundcheck: CH3 `SNARE B`, CH9 `GUITAR L`, CH10 `GUITAR R`;
  all detected signal and engine reached `running`.
- Measured snapshot:
  - CH3: peak `-6.3 dB`, RMS `-49.3 dB`, LUFS `-31.0`;
  - CH9: peak `-12.3 dB`, RMS `-30.1 dB`, LUFS `-25.5`;
  - CH10: peak `-11.8 dB`, RMS `-30.9 dB`, LUFS `-25.5`.
- Runtime authority: `proposal_only=true`, `auto_apply=false`, `dry_run=true`,
  `live_apply_enabled=false`.

## Оценка

| Область | Балл |
|---|---:|
| Live-safety и единая authority boundary | 19/20 |
| Рабочий WING/audio soundcheck runtime | 18/20 |
| Physical readback и transaction rollback | 18/20 |
| Тесты, CI, frontend/E2E | 15/15 |
| Сетевой периметр и аутентификация | 10/10 |
| Operations, fault proof, owner handoff | 12/15 |
| **Итого** | **92/100** |

## Открытые границы

- M005 cable-loss/timeout/stale-state proof требует физического действия
  оператора; программно он не имитировался.
- M006 owner readiness review остается отдельным подтверждением.
- Live authority не расширена на EQ, dynamics, gate, pan, delay, polarity,
  sends, DCA, routing, scenes, FX или master.
- Оценка относится к supervised soundcheck scope, а не к Full Autonomous
  Mixing Mode.
