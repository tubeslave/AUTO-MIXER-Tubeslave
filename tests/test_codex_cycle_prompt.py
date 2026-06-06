import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "tools" / "codex_autoloop" / "codex_cycle_prompt.md"
VALIDATOR_PATH = PROJECT_ROOT / "tools" / "codex_autoloop" / "validate_cycle_report.py"


def load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_cycle_report", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
        "set_trim",
        "set_input_trim",
        "set_channel_trim",
        "set_preamp_trim",
        "set_preamp_gain",
        "set_eq",
        "set_compressor",
        "route_output",
        "load_snap",
        "save_snap",
        "reset",
        "mute",
        "unmute",
        "set_insert",
        "set_fx_model",
        "set_fx_on",
        "set_fx_mix",
        "set_fx_parameter",
    ]

    assert "Не выполняй live OSC/WING writes." in text
    assert "Все опасные действия только dry-run/audit." in text
    assert "Не делай `git commit`, `git push`, `git reset --hard`" in text
    assert "Не трогай `.env`, токены, credentials, API keys и локальные секреты." in text
    for command_token in forbidden_command_tokens:
        assert f"`{command_token}`" in text


def test_codex_cycle_prompt_forbidden_commands_match_report_validator():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    validator = load_validator_module()

    for command_token in validator.FORBIDDEN_LIVE_COMMANDS:
        assert f"`{command_token}`" in text


def test_codex_cycle_prompt_requires_complete_human_report_sections():
    text = PROMPT_PATH.read_text(encoding="utf-8")

    assert "`automixer_codex_cycle_YYYYMMDDTHHMMSSZ.md`" in text
    assert "`Codex CLI was the only agent workflow used.`" in text
    assert "`External Paperclip/PiperClip was not used as an agent runner.`" in text
    assert "статус live safety: обязательно укажи, что live OSC/WING writes не выполнялись." in text
    assert "результат memory lint в отдельном разделе `## Memory Lint Result`" in text
    assert "`python3 tools/memory/memory_lint.py`" in text
    assert "результат memory lint" in text
    assert "`## Три обязательных вопроса`" in text
    assert "`## Объяснение простым языком`" in text
    assert "финальный top-level раздел" in text
    assert "Не добавляй после этого раздела другие top-level разделы." in text
    assert "### Что было сделано" in text
    assert "### Зачем это нужно проекту" in text
    assert "### В каком состоянии проект сейчас" in text
    assert "### Какие проблемы были обнаружены" in text
    assert "### Что нужно сделать дальше" in text
    assert "Не более 5 пунктов, по важности." in text
    assert "### Общая готовность направления" in text
    assert "Исследование: N%" in text
    assert "Готовность к реальному концерту: N%" in text
    assert "## 7. Три обязательных вопроса" in text
    assert "Перед финальным разделом `## Объяснение простым языком`" in text
    assert "1. Что нового проект научился делать?" in text
    assert "2. Что сейчас является самым слабым местом?" in text
    assert "3. Что даст наибольший прогресс в следующем цикле?" in text


def test_codex_cycle_prompt_matches_validator_end_section_order():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    validator = load_validator_module()

    questions_heading_index = text.index("## 7. Три обязательных вопроса")
    plain_explanation_index = text.index("## 8. Объяснение простым языком")

    assert questions_heading_index < plain_explanation_index
    assert validator.REQUIRED_END_SECTION_ORDER[0] == "## Три обязательных вопроса"
    assert validator.REQUIRED_END_SECTION_ORDER[4] == "## Объяснение простым языком"
    assert validator.PLAIN_EXPLANATION_HEADING == "## Объяснение простым языком"


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
        "`**/node_modules/`",
        "`backend/venv/`",
        "`**/.venv/`",
        "`artifacts/`",
        "`frontend/dist/`",
        "`logs/codex_autoloop/`",
        "`.paperclip/reports/`",
        "`Docs/archive/`",
        "`Docs/manuals/`",
        "`Docs/Behringer_WING_Official/`",
    ]

    for path in excluded_noise_paths:
        assert path in text

    assert "`rg --hidden --glob ...`" in text
    assert "`find ... | xargs rg`" in text
    assert "пробелы в путях" in text
