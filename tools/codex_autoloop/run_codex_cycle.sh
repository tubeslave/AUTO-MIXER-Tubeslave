#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPECTED_PROJECT_ROOT="${CODEX_AUTOLOOP_EXPECTED_PROJECT_ROOT:-/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main}"
PROMPT_FILE="${SCRIPT_DIR}/codex_cycle_prompt.md"
VALIDATOR="${SCRIPT_DIR}/validate_cycle_report.py"
BUDGET_CHECKER="${SCRIPT_DIR}/check_model_budget.py"
BUDGET_PROFILE="${SCRIPT_DIR}/model_budget_profile.json"
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

cleanup_stale_lock() {
  if [ ! -r "${LOCK_OWNER}" ]; then
    return 1
  fi
  LOCK_PID="$(awk -F= '/^pid=/{print $2; exit}' "${LOCK_OWNER}" 2>/dev/null || true)"
  if [ -z "${LOCK_PID}" ]; then
    return 1
  fi
  case "${LOCK_PID}" in
    *[!0-9]*)
      echo "Ignoring stale-lock cleanup because owner pid is invalid: ${LOCK_PID}."
      return 1
      ;;
  esac
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

if [ ! -r "${BUDGET_CHECKER}" ]; then
  echo "Model budget checker is missing or unreadable: ${BUDGET_CHECKER}"
  exit 66
fi

if [ ! -r "${BUDGET_PROFILE}" ]; then
  echo "Model budget profile is missing or unreadable: ${BUDGET_PROFILE}"
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
  BUDGET_ENV="${LOG_DIR}/codex_cycle_${STAMP}.budget.env"
  python3 "${BUDGET_CHECKER}" \
    --profile-config "${BUDGET_PROFILE}" \
    --task "${CODEX_AUTOLOOP_TASK_HINT:-supervised WING Rack MVP blocker}" \
    --format env \
    --output "${BUDGET_ENV}"
  # shellcheck disable=SC1090
  . "${BUDGET_ENV}"
  echo "Dry run enabled. Codex exec will not be started."
  echo "codex_path=${CODEX_PATH}"
  echo "selected_profile=${CODEX_AUTOLOOP_SELECTED_PROFILE:-skip}"
  echo "selected_model=${CODEX_AUTOLOOP_SELECTED_MODEL:-none}"
  echo "selected_reasoning=${CODEX_AUTOLOOP_SELECTED_REASONING_EFFORT:-none}"
  echo "Would run:"
  printf '  %q exec -C %q --sandbox workspace-write --model %q -c %q -c %q --color never --output-last-message %q - < %q\n' \
    "${CODEX_PATH}" \
    "${PROJECT_ROOT}" \
    "${CODEX_AUTOLOOP_SELECTED_MODEL:-from_config}" \
    'approval_policy="never"' \
    "model_reasoning_effort=\"${CODEX_AUTOLOOP_SELECTED_REASONING_EFFORT:-low}\"" \
    "${LAST_MESSAGE}" \
    "${PROMPT_FILE}"
  printf 'Would validate the new cycle report with: python3 %q <report>\n' "${VALIDATOR}"
  exit 0
fi

BUDGET_ENV="${LOG_DIR}/codex_cycle_${STAMP}.budget.env"
python3 "${BUDGET_CHECKER}" \
  --profile-config "${BUDGET_PROFILE}" \
  --task "${CODEX_AUTOLOOP_TASK_HINT:-supervised WING Rack MVP blocker}" \
  --reserve \
  --format env \
  --output "${BUDGET_ENV}"
# shellcheck disable=SC1090
. "${BUDGET_ENV}"

if [ "${CODEX_AUTOLOOP_SHOULD_RUN:-0}" != "1" ]; then
  echo "${CODEX_AUTOLOOP_SKIP_REASON:-Model budget gate skipped this cycle.}"
  echo "budget_state=${CODEX_AUTOLOOP_BUDGET_STATE:-unknown}"
  exit 0
fi

echo "selected_profile=${CODEX_AUTOLOOP_SELECTED_PROFILE}"
echo "selected_model=${CODEX_AUTOLOOP_SELECTED_MODEL}"
echo "selected_reasoning=${CODEX_AUTOLOOP_SELECTED_REASONING_EFFORT}"
echo "budget_state=${CODEX_AUTOLOOP_BUDGET_STATE}"

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
  --model "${CODEX_AUTOLOOP_SELECTED_MODEL}" \
  -c 'approval_policy="never"' \
  -c "model_reasoning_effort=\"${CODEX_AUTOLOOP_SELECTED_REASONING_EFFORT}\"" \
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
