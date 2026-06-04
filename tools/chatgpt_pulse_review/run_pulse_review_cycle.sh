#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPECTED_PROJECT_ROOT="${PULSE_REVIEW_EXPECTED_PROJECT_ROOT:-/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main}"
PROMPT_FILE="${SCRIPT_DIR}/pulse_review_prompt.md"
FETCHER="${SCRIPT_DIR}/fetch_chatgpt_pulse.py"
LOG_DIR="${PROJECT_ROOT}/logs/chatgpt_pulse_review"
REPORT_DIR="${PROJECT_ROOT}/.paperclip/reports"
LOCK_DIR="${PROJECT_ROOT}/.codex_autoloop.lock"
LOCK_OWNER="${LOCK_DIR}/owner.env"
STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
RUN_LOG="${LOG_DIR}/pulse_review_${STAMP}.log"
PULSE_JSON="${LOG_DIR}/pulse_${STAMP}.json"
PULSE_MD="${LOG_DIR}/pulse_${STAMP}.md"
EXPANDED_PROMPT="${LOG_DIR}/pulse_review_prompt_${STAMP}.md"
LAST_MESSAGE="${LOG_DIR}/pulse_review_${STAMP}.last.md"

mkdir -p "${LOG_DIR}" "${REPORT_DIR}"
exec >> "${RUN_LOG}" 2>&1

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] chatgpt pulse review: starting"
echo "project_root=${PROJECT_ROOT}"
echo "prompt_file=${PROMPT_FILE}"
echo "run_log=${RUN_LOG}"

if [ "${PROJECT_ROOT}" != "${EXPECTED_PROJECT_ROOT}" ]; then
  echo "Refusing to run outside the Automixer project."
  echo "expected_project_root=${EXPECTED_PROJECT_ROOT}"
  echo "actual_project_root=${PROJECT_ROOT}"
  exit 78
fi

GIT_ROOT="$(git -C "${PROJECT_ROOT}" rev-parse --show-toplevel 2>/dev/null || true)"
if [ "${GIT_ROOT}" != "${EXPECTED_PROJECT_ROOT}" ]; then
  echo "Refusing to run because git root is not the Automixer project."
  echo "expected_git_root=${EXPECTED_PROJECT_ROOT}"
  echo "actual_git_root=${GIT_ROOT:-not_a_git_repo}"
  exit 78
fi

find_codex() {
  if [ -n "${CODEX_BIN:-}" ] && [ -x "${CODEX_BIN}" ]; then
    printf '%s\n' "${CODEX_BIN}"
    return 0
  fi

  if command -v codex >/dev/null 2>&1; then
    command -v codex
    return 0
  fi

  if [ -d "${HOME}/.nvm/versions/node" ]; then
    find "${HOME}/.nvm/versions/node" -path "*/bin/codex" -type f -perm -111 2>/dev/null | sort | tail -n 1
    return 0
  fi

  return 1
}

find_python() {
  if [ -n "${PULSE_REVIEW_PYTHON:-}" ] && [ -x "${PULSE_REVIEW_PYTHON}" ]; then
    printf '%s\n' "${PULSE_REVIEW_PYTHON}"
    return 0
  fi

  LOCAL_HELPER_PY="${HOME}/.codex/skills/chatgpt-list-chats/.venv/bin/python"
  if [ -x "${LOCAL_HELPER_PY}" ]; then
    printf '%s\n' "${LOCAL_HELPER_PY}"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  return 1
}

CODEX_PATH="$(find_codex || true)"
PYTHON_PATH="$(find_python || true)"

if [ "${PULSE_REVIEW_DRY_RUN:-0}" = "1" ]; then
  echo "Dry run enabled. Pulse fetch and Codex exec will not be started."
  echo "codex_path=${CODEX_PATH:-codex_not_found}"
  echo "python_path=${PYTHON_PATH:-python_not_found}"
  echo "Would fetch Pulse to: ${PULSE_JSON}"
  echo "Would run Codex with prompt: ${EXPANDED_PROMPT}"
  exit 0
fi

cleanup_stale_lock() {
  if [ ! -r "${LOCK_OWNER}" ]; then
    return 1
  fi
  LOCK_PID="$(awk -F= '/^pid=/{print $2; exit}' "${LOCK_OWNER}" 2>/dev/null || true)"
  if [ -z "${LOCK_PID}" ]; then
    return 1
  fi
  if kill -0 "${LOCK_PID}" 2>/dev/null; then
    return 1
  fi
  echo "Removing stale automation lock owned by dead pid=${LOCK_PID}."
  rm -f "${LOCK_OWNER}" 2>/dev/null || true
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  cleanup_stale_lock || true
  if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "Another Codex automation run is already active: ${LOCK_DIR}"
    if [ -r "${LOCK_OWNER}" ]; then
      echo "Existing lock owner metadata:"
      while IFS= read -r line; do
        printf '  %s\n' "${line}"
      done < "${LOCK_OWNER}"
    fi
    exit 0
  fi
fi

cleanup() {
  rm -f "${LOCK_OWNER}" 2>/dev/null || true
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

{
  printf 'pid=%s\n' "$$"
  printf 'ppid=%s\n' "${PPID:-unknown}"
  printf 'started_at=%s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf 'stamp=%s\n' "${STAMP}"
  printf 'job=%s\n' "chatgpt_pulse_review"
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'run_log=%s\n' "${RUN_LOG}"
} > "${LOCK_OWNER}"

if [ ! -r "${PROMPT_FILE}" ]; then
  echo "Prompt file is missing or unreadable: ${PROMPT_FILE}"
  exit 66
fi

if [ ! -r "${FETCHER}" ]; then
  echo "Pulse fetcher is missing or unreadable: ${FETCHER}"
  exit 66
fi

if [ -z "${PYTHON_PATH}" ] || [ ! -x "${PYTHON_PATH}" ]; then
  echo "python executable was not found. Set PULSE_REVIEW_PYTHON."
  exit 127
fi

if [ -z "${CODEX_PATH}" ] || [ ! -x "${CODEX_PATH}" ]; then
  echo "codex executable was not found. Set CODEX_BIN or install Codex CLI."
  exit 127
fi

echo "python_path=${PYTHON_PATH}"
echo "codex_path=${CODEX_PATH}"
"${CODEX_PATH}" --version || true

cd "${PROJECT_ROOT}"

"${PYTHON_PATH}" "${FETCHER}" \
  --browser "${PULSE_REVIEW_BROWSER:-chrome}" \
  --output-json "${PULSE_JSON}" \
  --output-md "${PULSE_MD}"

cp "${PULSE_JSON}" "${LOG_DIR}/latest_pulse.json"
cp "${PULSE_MD}" "${LOG_DIR}/latest_pulse.md"

{
  cat "${PROMPT_FILE}"
  printf '\n\n## Current Pulse File\n\n'
  printf 'PULSE_JSON=%s\n' "${PULSE_JSON}"
  printf 'PULSE_MARKDOWN=%s\n' "${PULSE_MD}"
  printf 'RUN_LOG=%s\n' "${RUN_LOG}"
} > "${EXPANDED_PROMPT}"

CODEX_EXIT=0
"${CODEX_PATH}" exec \
  -C "${PROJECT_ROOT}" \
  --sandbox workspace-write \
  -c 'approval_policy="never"' \
  --color never \
  --output-last-message "${LAST_MESSAGE}" \
  - < "${EXPANDED_PROMPT}" || CODEX_EXIT=$?

if [ "${CODEX_EXIT}" -ne 0 ]; then
  echo "codex exec failed with exit code ${CODEX_EXIT}."
  exit "${CODEX_EXIT}"
fi

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] chatgpt pulse review: finished"
echo "pulse_json=${PULSE_JSON}"
echo "pulse_markdown=${PULSE_MD}"
echo "last_message=${LAST_MESSAGE}"
