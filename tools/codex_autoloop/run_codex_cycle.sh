#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPECTED_PROJECT_ROOT="${CODEX_AUTOLOOP_EXPECTED_PROJECT_ROOT:-/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main}"
PROMPT_FILE="${SCRIPT_DIR}/codex_cycle_prompt.md"
VALIDATOR="${SCRIPT_DIR}/validate_cycle_report.py"
LOG_DIR="${PROJECT_ROOT}/logs/codex_autoloop"
REPORT_DIR="${PROJECT_ROOT}/.paperclip/reports"
LOCK_DIR="${PROJECT_ROOT}/.codex_autoloop.lock"
LOCK_OWNER="${LOCK_DIR}/owner.env"
STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
RUN_LOG="${LOG_DIR}/codex_cycle_${STAMP}.log"
LAST_MESSAGE="${LOG_DIR}/codex_cycle_${STAMP}.last.md"
REPORT_MARKER="${LOG_DIR}/codex_cycle_${STAMP}.start"

mkdir -p "${LOG_DIR}" "${REPORT_DIR}"
exec >> "${RUN_LOG}" 2>&1

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] codex autoloop: starting"
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

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another codex autoloop run is already active: ${LOCK_DIR}"
  if [ -r "${LOCK_OWNER}" ]; then
    echo "Existing lock owner metadata:"
    while IFS= read -r line; do
      printf '  %s\n' "${line}"
    done < "${LOCK_OWNER}"
  else
    echo "Existing lock has no owner metadata."
  fi
  exit 0
fi

cleanup() {
  rm -f "${LOCK_OWNER}" 2>/dev/null || true
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

LOCK_HOST="$(hostname 2>/dev/null || printf 'unknown')"
{
  printf 'pid=%s\n' "$$"
  printf 'ppid=%s\n' "${PPID:-unknown}"
  printf 'started_at=%s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf 'stamp=%s\n' "${STAMP}"
  printf 'host=%s\n' "${LOCK_HOST}"
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'run_log=%s\n' "${RUN_LOG}"
} > "${LOCK_OWNER}"

if [ ! -r "${PROMPT_FILE}" ]; then
  echo "Prompt file is missing or unreadable: ${PROMPT_FILE}"
  exit 66
fi

if [ ! -r "${VALIDATOR}" ]; then
  echo "Cycle report validator is missing or unreadable: ${VALIDATOR}"
  exit 66
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

if [ "${CODEX_AUTOLOOP_DRY_RUN:-0}" = "1" ]; then
  CODEX_PATH="${CODEX_BIN:-$(command -v codex 2>/dev/null || printf 'codex')}"
  echo "Dry run enabled. Codex exec will not be started."
  echo "codex_path=${CODEX_PATH}"
  echo "Would run:"
  printf '  %q exec -C %q --sandbox workspace-write -c %q --color never --output-last-message %q - < %q\n' \
    "${CODEX_PATH}" \
    "${PROJECT_ROOT}" \
    'approval_policy="never"' \
    "${LAST_MESSAGE}" \
    "${PROMPT_FILE}"
  printf 'Would validate the new cycle report with: python3 %q <report>\n' "${VALIDATOR}"
  exit 0
fi

CODEX_PATH="$(find_codex || true)"
if [ -z "${CODEX_PATH}" ] || [ ! -x "${CODEX_PATH}" ]; then
  echo "codex executable was not found. Set CODEX_BIN or install Codex CLI."
  exit 127
fi

echo "codex_path=${CODEX_PATH}"
"${CODEX_PATH}" --version || true

cd "${PROJECT_ROOT}"

: > "${REPORT_MARKER}"
CODEX_EXIT=0
"${CODEX_PATH}" exec \
  -C "${PROJECT_ROOT}" \
  --sandbox workspace-write \
  -c 'approval_policy="never"' \
  --color never \
  --output-last-message "${LAST_MESSAGE}" \
  - < "${PROMPT_FILE}" || CODEX_EXIT=$?

if [ "${CODEX_EXIT}" -ne 0 ]; then
  echo "codex exec failed with exit code ${CODEX_EXIT}."
  exit "${CODEX_EXIT}"
fi

LATEST_REPORT="$(
  find "${REPORT_DIR}" -type f -name 'automixer_codex_cycle_*.md' -newer "${REPORT_MARKER}" -print \
    | sort \
    | tail -n 1
)"
if [ -z "${LATEST_REPORT}" ]; then
  echo "No new Codex cycle report found in ${REPORT_DIR}."
  exit 65
fi

python3 "${VALIDATOR}" "${LATEST_REPORT}"
echo "validated_report=${LATEST_REPORT}"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] codex autoloop: finished"
echo "last_message=${LAST_MESSAGE}"
