# AI Council Workspace

This directory is the shared memory layer between Codex.app, Codex CLI, Kimi CLI, and human maintainers.

## Layout

- `.ai/briefs/` - task briefs and definitions of done
- `.ai/proposals/` - independent proposals from Codex and Kimi
- `.ai/reviews/` - critiques, ADR reviews, and final patch reviews
- `.ai/decisions/` - optional short decision notes
- `.ai/memory/` - durable project memory
- `.ai/templates/` - reusable task templates
- `.ai/agent-core/` - shared agent rules, workflow, safety, source policy, and lint policy
- `.ai/project-layer/` - Automixer-specific goal, architecture, WING policy, safety, tasks, decisions, experiments, reports, and glossary
- `.ai/pipeline/` - Markdown memory pipeline, Codex/Paperclip cycles, and append templates

## Autonomy And Model Budget

- `.ai/agent-core/model_budget_policy.md` - virtual subagent roles and model
  escalation rules for autonomous Codex cycles
- `.ai/project-layer/autonomous_mvp_and_model_routing.md` - preserved full
  Automixer goal, current supervised WING MVP, and model routing strategy
- `tools/codex_autoloop/model_budget_profile.json` - local daily/weekly/monthly
  profile caps for GPT model use
- `tools/codex_autoloop/check_model_budget.py` - selects or skips one cycle
  before `codex exec` starts

## Project-Specific Notes

- Use `Docs/adr/`, not `docs/adr/`. This repository already uses `Docs/`, and keeping one canonical path avoids case-related confusion on macOS.
- As of `2026-04-22`, the active local branch is `master`, while the remote default branch points to `origin/main`.
- For council tasks, the safest base is the current checked-out branch unless the human explicitly asks for another base ref.

## Suggested Local Topology

- Primary repo: current checkout for planning, ADRs, and shared memory
- Codex worktree: sibling directory ending in `-codex`
- Kimi worktree: sibling directory ending in `-kimi`

## Standard Flow

1. Create `.ai/briefs/<task-id>.md`
2. Run `scripts/ai_council.sh proposal <task-id> codex`
3. Run `scripts/ai_council.sh proposal <task-id> kimi`
4. Run `scripts/ai_council.sh critique <task-id> codex`
5. Run `scripts/ai_council.sh critique <task-id> kimi`
6. Run `scripts/ai_council.sh adr <task-id>`
7. Implement in one worktree
8. Run `scripts/ai_council.sh patch-review <task-id> <writer-agent>`
9. Run `scripts/ai_council.sh lessons <task-id> codex`
10. Run `scripts/ai_council.sh lessons <task-id> kimi`

## Standard Test Command

`PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Memory Pipeline

The Markdown memory pipeline uses Agent Core + Project Layer + Pipeline:

1. Read `.ai/agent-core/` and `.ai/project-layer/` before planning.
2. Capture raw observations into `.ai/project-layer/experiments.md` or session reports.
3. Promote stable choices into `.ai/project-layer/decisions.md` or `Docs/adr/`.
4. Keep Paperclip-facing summaries in `.paperclip/reports/`.
5. Run `python3 tools/memory/memory_lint.py` after memory changes.

Memory files and tools are documentation or append-only helpers. They do not
grant permission for live OSC/WING writes.
