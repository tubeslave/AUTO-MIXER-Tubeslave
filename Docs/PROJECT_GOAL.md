# Цель проекта Automixer

## Авторитетная цель

Automixer должен стать профессиональным AI mixing engineer для live и studio
работы. Система должна анализировать многоканальное аудио, понимать источники и
роли в аранжировке, формировать безопасные решения по gain staging, балансу,
EQ, dynamics, de-essing, panning, frequency separation, unmasking, phase,
effects, sends, master bus и адаптации во время выступления.

Итоговая автономия сохраняется как долгосрочная North Star, но она не является
одной задачей реализации. Исполняемая единица разработки всегда должна быть
конечным milestone с независимой проверкой, rollback и evidence.

## Текущий приоритет

Текущий проверенный приоритет репозитория уже сужен до supervised WING Rack MVP.
Это не отменяет долгосрочную цель, но запрещает перескакивать к широкой live
автономии без доказанной безопасной hardware-петли.

MVP закрыт только когда есть реальные, принятые evidence artifacts для:

1. WING read-only connection proof.
2. Physical console readback proof.
3. Supervised fader write proof with explicit operator approval.
4. Supervised gain-down / trim proof with explicit operator approval.
5. Rollback proof.
6. Emergency stop proof.
7. Connection-loss / timeout fail-closed proof.
8. Operator-visible evidence report.
9. Readiness matrix update based only on real proof.

## Неприкосновенные границы

- Offline, shadow, replay, mock и simulator evidence не являются hardware proof.
- Full Autonomous Mixing Mode не может быть объявлен готовым по документации,
  отчету, зеленому unit test или LLM statement.
- Любая live WING запись до отдельного ADR и hardware proof ограничена только
  supervised `fader`/`gain` rail с явным approval, readback и rollback.
- EQ, compressor, gate, routing, mute, FX, scenes, matrices, DCA и full auto
  live writes остаются запрещенными для unattended automation.

## Источники текущей цели

- `.ai/project-layer/automixer_goal.md`
- `.ai/project-layer/autonomous_mvp_and_model_routing.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `backend/project_goal_capability_matrix.py`
- `backend/operator_mode_policy.py`

