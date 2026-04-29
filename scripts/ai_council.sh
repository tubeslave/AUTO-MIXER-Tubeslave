#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PARENT_DIR="$(cd "$(dirname "$REPO_ROOT")" && pwd)"
REPO_NAME="$(basename "$REPO_ROOT")"

DEFAULT_BASE_REF="$(git symbolic-ref --quiet --short HEAD || git rev-parse --short HEAD)"
DEFAULT_CODEX_BRANCH="ai/codex"
DEFAULT_KIMI_BRANCH="ai/kimi"
DEFAULT_CODEX_DIR="${PARENT_DIR}/${REPO_NAME}-codex"
DEFAULT_KIMI_DIR="${PARENT_DIR}/${REPO_NAME}-kimi"

brief_path() {
  printf '%s/.ai/briefs/%s.md' "$REPO_ROOT" "$1"
}

proposal_path() {
  printf '%s/.ai/proposals/%s.%s.md' "$REPO_ROOT" "$1" "$2"
}

critique_path() {
  printf '%s/.ai/reviews/%s.%s-on-%s.md' "$REPO_ROOT" "$1" "$2" "$3"
}

adr_path() {
  printf '%s/Docs/adr/%s.md' "$REPO_ROOT" "$1"
}

adr_review_path() {
  printf '%s/.ai/reviews/%s.%s-adr-review.md' "$REPO_ROOT" "$1" "$2"
}

patch_path() {
  printf '%s/.ai/reviews/%s.%s.patch' "$REPO_ROOT" "$1" "$2"
}

final_review_path() {
  printf '%s/.ai/reviews/%s.%s-final-review.md' "$REPO_ROOT" "$1" "$2"
}

lessons_path() {
  printf '%s/.ai/memory/%s.%s-lessons.md' "$REPO_ROOT" "$1" "$2"
}

ensure_layout() {
  mkdir -p \
    "$REPO_ROOT/.ai/briefs" \
    "$REPO_ROOT/.ai/proposals" \
    "$REPO_ROOT/.ai/reviews" \
    "$REPO_ROOT/.ai/decisions" \
    "$REPO_ROOT/.ai/memory" \
    "$REPO_ROOT/.ai/templates" \
    "$REPO_ROOT/Docs/adr"
}

usage() {
  cat <<EOF
Usage: scripts/ai_council.sh <command> [args]

Commands:
  bootstrap [base-ref]
      Create sibling worktrees for Codex and Kimi.

  scaffold-task <task-id> <title>
      Create .ai/briefs/<task-id>.md from the template.

  proposal <task-id> <codex|kimi>
      Generate an independent proposal.

  critique <task-id> <codex|kimi>
      Critique the other agent's proposal.

  adr <task-id>
      Synthesize proposals and critiques into Docs/adr/<task-id>.md using Codex.

  adr-review <task-id> [codex|kimi]
      Review the ADR for contradictions or missing risks. Defaults to kimi.

  patch-review <task-id> <codex|kimi> [base-ref]
      Export the writer's patch from its worktree and ask the other agent to review it.

  lessons <task-id> <codex|kimi>
      Generate durable lessons for .ai/memory/project.md.

  status
      Print current council paths, branches, and worktrees.
EOF
}

require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'Missing required file: %s\n' "$1" >&2
    exit 1
  fi
}

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

other_agent() {
  case "$1" in
    codex) printf 'kimi\n' ;;
    kimi) printf 'codex\n' ;;
    *)
      printf 'Unsupported agent: %s\n' "$1" >&2
      exit 1
      ;;
  esac
}

agent_label() {
  case "$1" in
    codex) printf 'Codex\n' ;;
    kimi) printf 'Kimi\n' ;;
    *)
      printf 'Unsupported agent: %s\n' "$1" >&2
      exit 1
      ;;
  esac
}

worktree_dir_for() {
  case "$1" in
    codex) printf '%s\n' "$DEFAULT_CODEX_DIR" ;;
    kimi) printf '%s\n' "$DEFAULT_KIMI_DIR" ;;
    *)
      printf 'Unsupported agent: %s\n' "$1" >&2
      exit 1
      ;;
  esac
}

run_codex_to_file() {
  local output_path="$1"
  local prompt="$2"

  require_tool codex
  codex exec --ephemeral -C "$REPO_ROOT" -o "$output_path" "$prompt"
}

run_kimi_to_file() {
  local output_path="$1"
  local prompt="$2"

  require_tool kimi
  kimi --plan --print --final-message-only --work-dir "$REPO_ROOT" --prompt "$prompt" > "$output_path"
}

run_agent_to_file() {
  local agent="$1"
  local output_path="$2"
  local prompt="$3"

  case "$agent" in
    codex) run_codex_to_file "$output_path" "$prompt" ;;
    kimi) run_kimi_to_file "$output_path" "$prompt" ;;
    *)
      printf 'Unsupported agent: %s\n' "$agent" >&2
      exit 1
      ;;
  esac
}

proposal_prompt() {
  local task_id="$1"
  local agent="$2"
  local other
  other="$(other_agent "$agent")"

  cat <<EOF
Read these files carefully:
- AGENTS.md
- CLAUDE.md
- .ai/briefs/${task_id}.md
- .ai/memory/project.md

You are $(agent_label "$agent") in the AI council for AUTO-MIXER-Tubeslave-main.
Do not edit files.
Return only the filled proposal using the proposal format from AGENTS.md.
Be concrete: cite likely files, risks, tests, and where $(agent_label "$other") may disagree.
EOF
}

critique_prompt() {
  local task_id="$1"
  local agent="$2"
  local other
  other="$(other_agent "$agent")"

  cat <<EOF
Read these files carefully:
- AGENTS.md
- CLAUDE.md
- .ai/briefs/${task_id}.md
- .ai/proposals/${task_id}.codex.md
- .ai/proposals/${task_id}.kimi.md
- .ai/memory/project.md

You are $(agent_label "$agent") in the AI council for AUTO-MIXER-Tubeslave-main.
Critique $(agent_label "$other")'s proposal using the critique format from AGENTS.md.
Steelman the other proposal first, then criticize it with concrete technical arguments.
End with your revised recommendation.
Do not edit files.
Return only the critique content.
EOF
}

adr_prompt() {
  local task_id="$1"

  cat <<EOF
Read these files carefully:
- AGENTS.md
- CLAUDE.md
- .ai/briefs/${task_id}.md
- .ai/proposals/${task_id}.codex.md
- .ai/proposals/${task_id}.kimi.md
- .ai/reviews/${task_id}.codex-on-kimi.md
- .ai/reviews/${task_id}.kimi-on-codex.md
- .ai/memory/project.md

Create an ADR for this task.
Return only Markdown, with no code fences and no extra commentary.
Use this structure exactly:

# ADR: <title>
## Context
## Options considered
## Decision
## Why this won
## Rejected alternatives
## Implementation plan
## Test plan
## Risks and rollback
EOF
}

adr_review_prompt() {
  local task_id="$1"
  local agent="$2"

  cat <<EOF
Read these files carefully:
- AGENTS.md
- CLAUDE.md
- .ai/briefs/${task_id}.md
- .ai/proposals/${task_id}.codex.md
- .ai/proposals/${task_id}.kimi.md
- .ai/reviews/${task_id}.codex-on-kimi.md
- .ai/reviews/${task_id}.kimi-on-codex.md
- Docs/adr/${task_id}.md

You are $(agent_label "$agent") in the AI council for AUTO-MIXER-Tubeslave-main.
Review the ADR against the proposals and critiques.
Look for contradictions, missing risks, weak rollback planning, and missing test coverage.
Do not edit files.
Return only the review.
EOF
}

patch_review_prompt() {
  local task_id="$1"
  local reviewer="$2"
  local writer="$3"

  cat <<EOF
Read these files carefully:
- AGENTS.md
- CLAUDE.md
- .ai/briefs/${task_id}.md
- Docs/adr/${task_id}.md
- .ai/reviews/${task_id}.${writer}.patch

You are $(agent_label "$reviewer") and the opposing reviewer for AUTO-MIXER-Tubeslave-main.
Review the patch against the brief and ADR.
Look for regressions, overengineering, missing tests, safety issues, and mismatches with the ADR.
Return one of: approve, request changes, or reject.
Provide concrete reasons with file-level specificity when possible.
Do not edit files.
Return only the review.
EOF
}

lessons_prompt() {
  local task_id="$1"
  local agent="$2"

  cat <<EOF
Read these files carefully:
- AGENTS.md
- CLAUDE.md
- Docs/adr/${task_id}.md
- .ai/reviews/${task_id}.codex-on-kimi.md
- .ai/reviews/${task_id}.kimi-on-codex.md
- .ai/memory/project.md

You are $(agent_label "$agent") in the AI council for AUTO-MIXER-Tubeslave-main.
Suggest 3-7 durable lessons for .ai/memory/project.md.
Do not include temporary task details.
Say whether each lesson belongs in AGENTS.md, .ai/memory/project.md, or another permanent doc.
Do not edit files.
Return only the lessons.
EOF
}

create_worktree() {
  local dir="$1"
  local branch="$2"
  local base_ref="$3"

  if git worktree list --porcelain | grep -Fx "worktree ${dir}" >/dev/null 2>&1; then
    printf 'Worktree already exists: %s\n' "$dir"
    return 0
  fi

  if [[ -e "$dir" && ! -d "$dir/.git" && ! -f "$dir/.git" ]]; then
    printf 'Path exists and is not a git worktree: %s\n' "$dir" >&2
    exit 1
  fi

  if git show-ref --verify --quiet "refs/heads/${branch}"; then
    git worktree add "$dir" "$branch"
  else
    git worktree add "$dir" -b "$branch" "$base_ref"
  fi
}

cmd_bootstrap() {
  local base_ref="${1:-$DEFAULT_BASE_REF}"

  ensure_layout
  create_worktree "$DEFAULT_CODEX_DIR" "$DEFAULT_CODEX_BRANCH" "$base_ref"
  create_worktree "$DEFAULT_KIMI_DIR" "$DEFAULT_KIMI_BRANCH" "$base_ref"

  cat <<EOF
Council bootstrap complete.

Base ref:    ${base_ref}
Codex tree:  ${DEFAULT_CODEX_DIR} (${DEFAULT_CODEX_BRANCH})
Kimi tree:   ${DEFAULT_KIMI_DIR} (${DEFAULT_KIMI_BRANCH})
EOF
}

cmd_scaffold_task() {
  local task_id="${1:-}"
  local title="${2:-}"
  local target

  if [[ -z "$task_id" || -z "$title" ]]; then
    printf 'Usage: scripts/ai_council.sh scaffold-task <task-id> <title>\n' >&2
    exit 1
  fi

  ensure_layout
  target="$(brief_path "$task_id")"
  if [[ -e "$target" ]]; then
    printf 'Task brief already exists: %s\n' "$target" >&2
    exit 1
  fi

  cat > "$target" <<EOF
# Task ${task_id}: ${title}

## Problem

Describe the concrete problem or refactor target.

## Constraints

- Preserve safety-critical behavior unless the brief explicitly changes it.
- Keep the public API stable unless the brief says otherwise.
- Avoid new production dependencies without a strong written reason.

## Definition of Done

- Tests pass
- Behavior matches the brief
- ADR is updated when architecture or behavior changes

## Test Command

\`PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q\`
EOF

  printf 'Created %s\n' "$target"
}

cmd_proposal() {
  local task_id="${1:-}"
  local agent="${2:-}"
  local output
  local prompt

  if [[ -z "$task_id" || -z "$agent" ]]; then
    printf 'Usage: scripts/ai_council.sh proposal <task-id> <codex|kimi>\n' >&2
    exit 1
  fi

  require_file "$(brief_path "$task_id")"
  require_file "$REPO_ROOT/AGENTS.md"
  require_file "$REPO_ROOT/CLAUDE.md"
  require_file "$REPO_ROOT/.ai/memory/project.md"

  output="$(proposal_path "$task_id" "$agent")"
  prompt="$(proposal_prompt "$task_id" "$agent")"
  run_agent_to_file "$agent" "$output" "$prompt"

  printf 'Wrote %s\n' "$output"
}

cmd_critique() {
  local task_id="${1:-}"
  local agent="${2:-}"
  local other
  local output
  local prompt

  if [[ -z "$task_id" || -z "$agent" ]]; then
    printf 'Usage: scripts/ai_council.sh critique <task-id> <codex|kimi>\n' >&2
    exit 1
  fi

  other="$(other_agent "$agent")"
  require_file "$(brief_path "$task_id")"
  require_file "$(proposal_path "$task_id" "codex")"
  require_file "$(proposal_path "$task_id" "kimi")"

  output="$(critique_path "$task_id" "$agent" "$other")"
  prompt="$(critique_prompt "$task_id" "$agent")"
  run_agent_to_file "$agent" "$output" "$prompt"

  printf 'Wrote %s\n' "$output"
}

cmd_adr() {
  local task_id="${1:-}"
  local output
  local prompt

  if [[ -z "$task_id" ]]; then
    printf 'Usage: scripts/ai_council.sh adr <task-id>\n' >&2
    exit 1
  fi

  require_file "$(brief_path "$task_id")"
  require_file "$(proposal_path "$task_id" "codex")"
  require_file "$(proposal_path "$task_id" "kimi")"
  require_file "$(critique_path "$task_id" "codex" "kimi")"
  require_file "$(critique_path "$task_id" "kimi" "codex")"

  output="$(adr_path "$task_id")"
  prompt="$(adr_prompt "$task_id")"
  run_codex_to_file "$output" "$prompt"

  printf 'Wrote %s\n' "$output"
}

cmd_adr_review() {
  local task_id="${1:-}"
  local agent="${2:-kimi}"
  local output
  local prompt

  if [[ -z "$task_id" ]]; then
    printf 'Usage: scripts/ai_council.sh adr-review <task-id> [codex|kimi]\n' >&2
    exit 1
  fi

  require_file "$(adr_path "$task_id")"
  output="$(adr_review_path "$task_id" "$agent")"
  prompt="$(adr_review_prompt "$task_id" "$agent")"
  run_agent_to_file "$agent" "$output" "$prompt"

  printf 'Wrote %s\n' "$output"
}

cmd_patch_review() {
  local task_id="${1:-}"
  local writer="${2:-}"
  local base_ref="${3:-$DEFAULT_BASE_REF}"
  local reviewer
  local worktree_dir
  local exported_patch
  local output
  local prompt

  if [[ -z "$task_id" || -z "$writer" ]]; then
    printf 'Usage: scripts/ai_council.sh patch-review <task-id> <codex|kimi> [base-ref]\n' >&2
    exit 1
  fi

  reviewer="$(other_agent "$writer")"
  worktree_dir="$(worktree_dir_for "$writer")"

  if [[ ! -d "$worktree_dir" ]]; then
    printf 'Writer worktree not found: %s\nRun bootstrap first.\n' "$worktree_dir" >&2
    exit 1
  fi

  require_file "$(brief_path "$task_id")"
  require_file "$(adr_path "$task_id")"

  exported_patch="$(patch_path "$task_id" "$writer")"
  git -C "$worktree_dir" diff "${base_ref}...HEAD" > "$exported_patch"

  output="$(final_review_path "$task_id" "$reviewer")"
  prompt="$(patch_review_prompt "$task_id" "$reviewer" "$writer")"
  run_agent_to_file "$reviewer" "$output" "$prompt"

  printf 'Wrote %s and %s\n' "$exported_patch" "$output"
}

cmd_lessons() {
  local task_id="${1:-}"
  local agent="${2:-}"
  local output
  local prompt

  if [[ -z "$task_id" || -z "$agent" ]]; then
    printf 'Usage: scripts/ai_council.sh lessons <task-id> <codex|kimi>\n' >&2
    exit 1
  fi

  require_file "$(adr_path "$task_id")"
  output="$(lessons_path "$task_id" "$agent")"
  prompt="$(lessons_prompt "$task_id" "$agent")"
  run_agent_to_file "$agent" "$output" "$prompt"

  printf 'Wrote %s\n' "$output"
}

cmd_status() {
  cat <<EOF
Repo root:   ${REPO_ROOT}
Base ref:    ${DEFAULT_BASE_REF}
Codex dir:   ${DEFAULT_CODEX_DIR}
Kimi dir:    ${DEFAULT_KIMI_DIR}
Codex branch:${DEFAULT_CODEX_BRANCH}
Kimi branch: ${DEFAULT_KIMI_BRANCH}

Current worktrees:
EOF
  git worktree list
}

main() {
  local command="${1:-}"

  if [[ -z "$command" ]]; then
    usage
    exit 1
  fi

  shift

  case "$command" in
    bootstrap) cmd_bootstrap "$@" ;;
    scaffold-task) cmd_scaffold_task "$@" ;;
    proposal) cmd_proposal "$@" ;;
    critique) cmd_critique "$@" ;;
    adr) cmd_adr "$@" ;;
    adr-review) cmd_adr_review "$@" ;;
    patch-review) cmd_patch_review "$@" ;;
    lessons) cmd_lessons "$@" ;;
    status) cmd_status ;;
    help|-h|--help) usage ;;
    *)
      printf 'Unknown command: %s\n\n' "$command" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
