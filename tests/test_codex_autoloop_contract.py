import plistlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTOLOOP_DIR = PROJECT_ROOT / "tools" / "codex_autoloop"
RUNNER_PATH = AUTOLOOP_DIR / "run_codex_cycle.sh"
VALIDATOR_PATH = AUTOLOOP_DIR / "validate_cycle_report.py"
PLIST_PATH = AUTOLOOP_DIR / "com.tubeslave.automixer.codex-autoloop.plist"
README_PATH = AUTOLOOP_DIR / "README.md"
PROMPT_PATH = AUTOLOOP_DIR / "codex_cycle_prompt.md"


def test_codex_autoloop_runner_uses_repo_prompt_and_one_shot_exec():
    text = RUNNER_PATH.read_text(encoding="utf-8")

    assert 'PROMPT_FILE="${SCRIPT_DIR}/codex_cycle_prompt.md"' in text
    assert 'VALIDATOR="${SCRIPT_DIR}/validate_cycle_report.py"' in text
    assert 'LOG_DIR="${PROJECT_ROOT}/logs/codex_autoloop"' in text
    assert 'REPORT_DIR="${PROJECT_ROOT}/.paperclip/reports"' in text
    assert 'LOCK_DIR="${PROJECT_ROOT}/.codex_autoloop.lock"' in text
    assert 'LOCK_OWNER="${LOCK_DIR}/owner.env"' in text
    assert 'REPORT_MARKER="${LOG_DIR}/codex_cycle_${STAMP}.start"' in text
    assert 'if ! mkdir "${LOCK_DIR}"' in text
    assert 'Another codex autoloop run is already active' in text
    assert 'Existing lock owner metadata:' in text
    assert 'Existing lock has no owner metadata.' in text
    assert 'case "${LOCK_PID}" in' in text
    assert '*[!0-9]*)' in text
    assert 'Ignoring stale-lock cleanup because owner pid is invalid' in text
    assert 'return 1' in text
    assert "printf 'pid=%s\\n' \"$$\"" in text
    assert "printf 'ppid=%s\\n' \"${PPID:-unknown}\"" in text
    assert "printf 'started_at=%s\\n' \"$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")\"" in text
    assert "printf 'stamp=%s\\n' \"${STAMP}\"" in text
    assert "printf 'host=%s\\n' \"${LOCK_HOST}\"" in text
    assert "printf 'project_root=%s\\n' \"${PROJECT_ROOT}\"" in text
    assert "printf 'run_log=%s\\n' \"${RUN_LOG}\"" in text
    assert 'rm -f "${LOCK_OWNER}" 2>/dev/null || true' in text
    assert 'rmdir "${LOCK_DIR}" 2>/dev/null || true' in text
    assert 'rm -rf "${LOCK_DIR}"' not in text
    assert 'exec >> "${RUN_LOG}" 2>&1' in text
    assert ">(tee" not in text

    assert '"${CODEX_PATH}" exec \\' in text
    assert '-C "${PROJECT_ROOT}"' in text
    assert "--sandbox workspace-write" in text
    assert '-c \'approval_policy="never"\'' in text
    assert '--output-last-message "${LAST_MESSAGE}"' in text
    assert '- < "${PROMPT_FILE}"' in text

    assert ': > "${REPORT_MARKER}"' in text
    assert 'find "${REPORT_DIR}" -type f -name \'automixer_codex_cycle_*.md\' -newer "${REPORT_MARKER}" -print \\' in text
    assert 'No new Codex cycle report found in ${REPORT_DIR}.' in text
    assert 'python3 "${VALIDATOR}" "${LATEST_REPORT}"' in text
    assert 'validated_report=${LATEST_REPORT}' in text
    assert text.index('"${CODEX_PATH}" exec \\') < text.index('python3 "${VALIDATOR}" "${LATEST_REPORT}"')
    assert text.index('python3 "${VALIDATOR}" "${LATEST_REPORT}"') < text.index("codex autoloop: finished")

    assert "while true" not in text
    assert "for ((;;" not in text
    assert "until true" not in text


def test_codex_autoloop_runner_fails_closed_before_codex_launch():
    text = RUNNER_PATH.read_text(encoding="utf-8")

    expected_root = (
        'EXPECTED_PROJECT_ROOT="${CODEX_AUTOLOOP_EXPECTED_PROJECT_ROOT:-'
        "/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main}\""
    )
    assert expected_root in text
    assert 'if [ "${PROJECT_ROOT}" != "${EXPECTED_PROJECT_ROOT}" ]; then' in text
    assert 'if [ "${GIT_ROOT}" != "${EXPECTED_PROJECT_ROOT}" ]; then' in text
    assert 'Prompt file is missing or unreadable: ${PROMPT_FILE}' in text
    assert 'Cycle report validator is missing or unreadable: ${VALIDATOR}' in text

    prompt_guard_index = text.index('if [ ! -r "${PROMPT_FILE}" ]; then')
    validator_guard_index = text.index('if [ ! -r "${VALIDATOR}" ]; then')
    dry_run_index = text.index('if [ "${CODEX_AUTOLOOP_DRY_RUN:-0}" = "1" ]; then')
    codex_lookup_index = text.index('CODEX_PATH="$(find_codex || true)"')
    codex_missing_index = text.index("codex executable was not found")
    codex_exec_index = text.index('"${CODEX_PATH}" exec \\')
    assert prompt_guard_index < codex_exec_index
    assert validator_guard_index < codex_exec_index
    assert dry_run_index < codex_lookup_index
    assert dry_run_index < codex_missing_index
    assert 'CODEX_PATH="${CODEX_BIN:-$(command -v codex 2>/dev/null || printf \'codex\')}"' in text


def test_codex_autoloop_launch_agent_is_interval_based_and_non_keepalive():
    with PLIST_PATH.open("rb") as plist_file:
        launch_agent = plistlib.load(plist_file)

    assert launch_agent["ProgramArguments"] == [
        "/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/tools/codex_autoloop/run_codex_cycle.sh"
    ]
    assert launch_agent["WorkingDirectory"] == "/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main"
    assert launch_agent["RunAtLoad"] is False
    assert launch_agent["KeepAlive"] is False
    assert launch_agent["StartInterval"] == 1200
    assert (
        launch_agent["EnvironmentVariables"]["CODEX_AUTOLOOP_EXPECTED_PROJECT_ROOT"]
        == "/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main"
    )


def test_codex_autoloop_stays_inside_automixer_and_repo_local_reports():
    runner_text = RUNNER_PATH.read_text(encoding="utf-8")
    plist_text = PLIST_PATH.read_text(encoding="utf-8")
    forbidden_external_paths = [
        "/Users/dmitrijvolkov/paperclip",
        "/Users/dmitrijvolkov/automixer-kimi-paperclip-analyst",
    ]

    assert 'REPORT_DIR="${PROJECT_ROOT}/.paperclip/reports"' in runner_text
    assert "/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main" in runner_text
    assert "/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main" in plist_text

    for forbidden_path in forbidden_external_paths:
        assert forbidden_path not in runner_text
        assert forbidden_path not in plist_text


def test_codex_cycle_prompt_requires_repo_local_safety_report_shape():
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")

    required_report_phrases = [
        "Выполни ровно один законченный безопасный цикл разработки и выйди.",
        "Не запускай бесконечный цикл.",
        "`/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/.paperclip/reports/`",
        "Не используй внешний Paperclip/PiperClip как систему запуска агентов.",
        "`Codex CLI was the only agent workflow used.`",
        "`External Paperclip/PiperClip was not used as an agent runner.`",
        "`Repo-local report path:`",
        "статус live safety: обязательно укажи, что live OSC/WING writes не выполнялись.",
        "`## Три обязательных вопроса`",
        "`## Объяснение простым языком`",
        "## 7. Три обязательных вопроса",
        "## 8. Объяснение простым языком",
        "Перед финальным разделом `## Объяснение простым языком`",
        "1. Что нового проект научился делать?",
        "2. Что сейчас является самым слабым местом?",
        "3. Что даст наибольший прогресс в следующем цикле?",
    ]

    for phrase in required_report_phrases:
        assert phrase in prompt_text


def test_codex_autoloop_readme_documents_exact_agent_scope_lines():
    readme_text = README_PATH.read_text(encoding="utf-8")

    assert "exact Scope lines" in readme_text
    assert "`Codex CLI was the only agent workflow used.`" in readme_text
    assert "`External Paperclip/PiperClip was not used as an agent runner.`" in readme_text
    assert "`Repo-local report path:`" in readme_text
    assert (
        "/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/.paperclip/reports"
        in readme_text
    )


def test_codex_autoloop_readme_documents_nested_dependency_todo_scan_noise():
    readme_text = README_PATH.read_text(encoding="utf-8")
    normalized_readme_text = " ".join(readme_text.split())

    assert "TODO/FIXME scans focused on owned Automixer code" in normalized_readme_text
    assert "`frontend/node_modules/`" in readme_text
    assert "`**/node_modules/`" in readme_text
    assert "`backend/venv/`" in readme_text
    assert "`**/.venv/`" in readme_text
    assert (
        "scan noise unless the selected task explicitly targets them"
        in normalized_readme_text
    )


def test_codex_autoloop_limit_gate_policy_matches_current_runner():
    runner_text = RUNNER_PATH.read_text(encoding="utf-8")
    plist_text = PLIST_PATH.read_text(encoding="utf-8")
    readme_text = README_PATH.read_text(encoding="utf-8")
    forbidden_limit_gate_tokens = [
        "CODEX_LIMITS_FILE",
        "CODEX_LIMIT_STATE_FILE",
        "CODEX_LIMITS_REQUIRED",
        "check_codex_limits.py",
        "codex_limits.example.json",
        "limit_state.json",
        "remaining_percent",
        "skip_unknown_limits",
    ]

    assert "Codex limit checks are disabled." in readme_text
    assert "does not skip runs based on limits" in readme_text
    assert "Current policy supersedes older local reports that described a limit gate" in readme_text
    assert not (AUTOLOOP_DIR / "check_codex_limits.py").exists()
    assert not (AUTOLOOP_DIR / "codex_limits.example.json").exists()
    assert VALIDATOR_PATH.exists()

    for token in forbidden_limit_gate_tokens:
        assert token not in runner_text
        assert token not in plist_text
