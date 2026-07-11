# Soundcheck Workflow 95

## Задача

Довести локально проверяемый режим саундчека до оценки не ниже 95/100, не
расширяя фактические полномочия записи в WING и не подменяя физические M005/M006
программными тестами.

## Исходные дефекты

- канонический web-handler запускал только анализ, а построение рекомендаций
  находилось внутри apply-ветки;
- CLI и конфигурация допускали обход единого supervised apply-gate;
- soundcheck engine создавал собственные mixer/audio services и мог подменять
  server runtime остановленными объектами;
- активный React frontend не имел рабочего soundcheck cockpit;
- отсутствовал long-run контракт абсолютных, идемпотентных рекомендаций.

## Граница работы

Разрешены локальные изменения кода, тесты и offline/no-write проверки. Запрещены
необслуживаемые обращения к реальному пульту и любые новые standing live-write
полномочия. Сброс обработки, повышение gain/trim и auto-apply остаются закрыты.

## Definition of done

1. Proposal-only run всегда строит reviewable recommendations при достаточных
   измерениях и не отправляет mixer writes.
2. Live apply невозможно включить через CLI или YAML в обход server gate.
3. Shared mixer/audio runtime не останавливается и не подменяется engine.
4. Активный frontend показывает настоящий backend status, этап, ограничения,
   ошибки и очередь предложений; start/stop работают через canonical handler.
5. Regression, lifecycle, long-run, frontend и full-suite gates проходят.
6. Локальный soundcheck workflow получает минимум 95/100 по опубликованной
   рубрике; live-release readiness оценивается отдельно и остается заблокирована
   до физического M005 и owner review M006.

