# First-fixes checklist (2–3 дня)

Цель: привести проект в состояние, где он снова проверяемом, документация
не противоречит реальной структуре, а runtime-артефакты не засоряют репозиторий.

---

## P0 — День 1. Сделать проект снова проверяемым

### [ ] 1. Исправить импорт-барьер в тестах
- **Проблема:** тесты падают на `import server` → `import pyaudio`, которого нет в текущем venv.
- **Действие:**
  ```bash
  cd /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main
  PYTHONPATH=backend ./backend/venv/bin/pip install pyaudio
  ./backend/venv/bin/python -m pytest tests/ -x --tb=short -q
  ```
- **Критерий:** хотя бы 1 тест проходит, сбор не падает на ImportError.

### [ ] 2. Очистить `.gitignore`
- **Проблема:** runtime-артефакты не явно исключены.
- **Действие:** добавить в `.gitignore`:
  ```gitignore
  # Runtime artifacts
  reports/
  sessions/
  .ai/
  offline_test_output/
  production_mix_v1_out/
  logs/
  *.pcap
  *.pcap.status.json
  ```

---

## P1 — День 2. Документация и запуск без противоречий

### [ ] 3. Привести `README.md` в соответствие с текущей архитектурой
- **Проблема:** README рисует более раннюю версию проекта, пути не совпадают.
- **Действие:**
  - Сделать README кратким “стартовым” руководством.
  - Путь к серверу: `PYTHONPATH=backend python backend/server.py`.
  - Убрать упоминания компонентов, которых нет/не на тех местах.
  - Добавить раздел “См. также: CLAUDE.md, INSTALL.md”.

### [ ] 4. Убрать дублирующие стартовые пути
- **Проблема:** несколько entrypoint-обёрток рисуют разные способы запуска.
- **Действие:**
  - `start_soundcheck.py` — оставить как primary CLI entrypoint.
  - `start_backend.sh` / `start_backend.bat` — сделать thin-wrapper:
    ```bash
    # start_backend.sh
    set -e
    cd "$(dirname "$0")"
    PYTHONPATH=backend python backend/server.py
    ```
  - README/CLAUDE: привести к одному источнику.

---

## P2 — День 3. Структура и подготовка к рефакторингу

### [ ] 5. Уточнить пакетную структуру `backend/`
- **Проблема:** 50+ модулей в плоском слое, сложнее рефакторить.
- **Действие:** создать `Docs/PACKAGE_MAP.md`:
  - Список групп модулей: `handlers/`, `agents/`, `ai/`, `ml/`, `osc/`.
  - Публичные vs внутренние API.
  - Это ускорит следующий спринт рефакторинга.

### [ ] 6. Отделить `mix_agent/` от runtime backend
- **Проблема:** `mix_agent/` позиционируется как offline facade, но лежит рядом с production backend.
- **Действие:** явно прописать в `CLAUDE.md`:
  - `mix_agent/` не трогает пульт.
  - Все команды к пульту проходят через `AutoFOHSafetyController`.
  - Запускать `mix_agent` запрещено при активном live-сеансе без явного opt-in флага.

---

## Итоговые критерии приёмки

- [ ] `pytest tests/ -x --tb=short -q` проходит минимум часть тестов, нет ImportError на старте.
- [ ] В репозитории нет артефактов из `reports/`, `sessions/`, `.ai/`, `offline_test_output/` в `git status`.
- [ ] Один canonical entrypoint для soundcheck, один — для backend.
- [ ] README соответствует реальной структуре.
- [ ] `Docs/PACKAGE_MAP.md` существует и содержит актуальную карту.
- [ ] В `CLAUDE.md` есть явное разделение `mix_agent/` и runtime backend.
