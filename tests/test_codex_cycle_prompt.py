from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "tools" / "codex_autoloop" / "codex_cycle_prompt.md"


def test_codex_cycle_prompt_keeps_cycles_bounded_to_automixer_repo():
    text = PROMPT_PATH.read_text(encoding="utf-8")

    assert "Выполни ровно один законченный безопасный цикл разработки и выйди." in text
    assert "Не запускай бесконечный цикл." in text
    assert "`/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main`" in text
    assert "`/Users/dmitrijvolkov/paperclip`" in text
    assert "`/Users/dmitrijvolkov/automixer-kimi-paperclip-analyst`" in text
    assert "`/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/.paperclip/reports/`" in text
    assert "Не используй внешний Paperclip/PiperClip как систему запуска агентов." in text


def test_codex_cycle_prompt_forbids_live_writes_and_destructive_actions():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    forbidden_command_tokens = [
        "set_fader",
        "set_gain",
        "set_eq",
        "set_compressor",
        "route_output",
        "load_snap",
        "save_snap",
        "reset",
        "mute",
    ]

    assert "Не выполняй live OSC/WING writes." in text
    assert "Все опасные действия только dry-run/audit." in text
    assert "Не делай `git commit`, `git push`, `git reset --hard`" in text
    assert "Не трогай `.env`, токены, credentials, API keys и локальные секреты." in text
    for command_token in forbidden_command_tokens:
        assert f"`{command_token}`" in text


def test_codex_cycle_prompt_requires_complete_human_report_sections():
    text = PROMPT_PATH.read_text(encoding="utf-8")

    assert "`automixer_codex_cycle_YYYYMMDDTHHMMSSZ.md`" in text
    assert "статус live safety: обязательно укажи, что live OSC/WING writes не выполнялись." in text
    assert "`python3 tools/memory/memory_lint.py`" in text
    assert "результат memory lint" in text
    assert "`## Объяснение простым языком`" in text
    assert "### Что было сделано" in text
    assert "### Зачем это нужно проекту" in text
    assert "### В каком состоянии проект сейчас" in text
    assert "### Какие проблемы были обнаружены" in text
    assert "### Что нужно сделать дальше" in text
    assert "Не более 5 пунктов, по важности." in text
    assert "### Общая готовность направления" in text
    assert "Исследование: N%" in text
    assert "Готовность к реальному концерту: N%" in text
    assert "## 8. Три обязательных вопроса" in text
    assert "1. Что нового проект научился делать?" in text
    assert "2. Что сейчас является самым слабым местом?" in text
    assert "3. Что даст наибольший прогресс в следующем цикле?" in text


def test_codex_cycle_prompt_keeps_todo_scan_focused_on_owned_project_paths():
    text = PROMPT_PATH.read_text(encoding="utf-8")

    assert "Для TODO/FIXME scan фокусируйся на owned коде" in text
    assert "excluded scan noise" in text
    excluded_noise_paths = [
        "`external/`",
        "`models/training_state.json`",
        "`backend/ai/knowledge/`",
        "`frontend/node_modules/`",
        "`node_modules/`",
        "`artifacts/`",
        "`Docs/archive/`",
        "`Docs/manuals/`",
        "`Docs/Behringer_WING_Official/`",
    ]

    for path in excluded_noise_paths:
        assert path in text
