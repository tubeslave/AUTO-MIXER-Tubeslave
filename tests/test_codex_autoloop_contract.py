import plistlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTOLOOP_DIR = PROJECT_ROOT / "tools" / "codex_autoloop"
RUNNER_PATH = AUTOLOOP_DIR / "run_codex_cycle.sh"
PLIST_PATH = AUTOLOOP_DIR / "com.tubeslave.automixer.codex-autoloop.plist"


def test_codex_autoloop_runner_uses_repo_prompt_and_one_shot_exec():
    text = RUNNER_PATH.read_text(encoding="utf-8")

    assert 'PROMPT_FILE="${SCRIPT_DIR}/codex_cycle_prompt.md"' in text
    assert 'LOG_DIR="${PROJECT_ROOT}/logs/codex_autoloop"' in text
    assert 'REPORT_DIR="${PROJECT_ROOT}/.paperclip/reports"' in text
    assert 'LOCK_DIR="${PROJECT_ROOT}/.codex_autoloop.lock"' in text
    assert 'if ! mkdir "${LOCK_DIR}"' in text
    assert 'Another codex autoloop run is already active' in text
    assert 'rmdir "${LOCK_DIR}" 2>/dev/null || true' in text
    assert 'rm -rf "${LOCK_DIR}"' not in text

    assert '"${CODEX_PATH}" exec \\' in text
    assert '-C "${PROJECT_ROOT}"' in text
    assert "--sandbox workspace-write" in text
    assert '-c \'approval_policy="never"\'' in text
    assert '--output-last-message "${LAST_MESSAGE}"' in text
    assert '- < "${PROMPT_FILE}"' in text

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

    prompt_guard_index = text.index('if [ ! -r "${PROMPT_FILE}" ]; then')
    codex_exec_index = text.index('"${CODEX_PATH}" exec \\')
    assert prompt_guard_index < codex_exec_index


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
