#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPECTED_PROJECT_ROOT="${CODEX_AUTOLOOP_EXPECTED_PROJECT_ROOT:-/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main}"
PROMPT_FILE="${SCRIPT_DIR}/codex_cycle_prompt.md"
LOG_DIR="${PROJECT_ROOT}/logs/codex_autoloop"
REPORT_DIR="${PROJECT_ROOT}/.paperclip/reports"
LOCK_DIR="${PROJECT_ROOT}/.codex_autoloop.lock"
STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
RUN_LOG="${LOG_DIR}/codex_cycle_${STAMP}.log"
LAST_MESSAGE="${LOG_DIR}/codex_cycle_${STAMP}.last.md"

mkdir -p "${LOG_DIR}" "${REPORT_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

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
  exit 0
fi

cleanup() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ ! -r "${PROMPT_FILE}" ]; then
  echo "Prompt file is missing or unreadable: ${PROMPT_FILE}"
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

CODEX_PATH="$(find_codex || true)"
if [ -z "${CODEX_PATH}" ] || [ ! -x "${CODEX_PATH}" ]; then
  echo "codex executable was not found. Set CODEX_BIN or install Codex CLI."
  exit 127
fi

echo "codex_path=${CODEX_PATH}"
"${CODEX_PATH}" --version || true

if [ "${CODEX_AUTOLOOP_DRY_RUN:-0}" = "1" ]; then
  echo "Dry run enabled. Codex exec will not be started."
  echo "Would run:"
  printf '  %q exec -C %q --sandbox workspace-write -c %q --color never --output-last-message %q - < %q\n' \
    "${CODEX_PATH}" \
    "${PROJECT_ROOT}" \
    'approval_policy="never"' \
    "${LAST_MESSAGE}" \
    "${PROMPT_FILE}"
  exit 0
fi

cd "${PROJECT_ROOT}"

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

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] codex autoloop: finished"
echo "last_message=${LAST_MESSAGE}"
