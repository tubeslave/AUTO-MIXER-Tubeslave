# Manual DSP Agent Task

- Created: 2026-05-08T14:32:24.407Z
- Reason: dryRun=true; Codex CLI was not invoked.
- Repository: /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main
- Branch: feature/mcp-director
- Task: task_20260508T143224Z_d3a621be
- Timeout seconds: 300

## Prompt

Analyze EQ decision logic in AUTO-MIXER-Tubeslave and create a dry-run report. Do not modify files.

## Manual Command

Run this from a terminal after reviewing branch state:

```bash
codex exec -C '/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main' -s read-only 'You are DSP Agent for AUTO-MIXER-Tubeslave.

Work in read-only mode. Do not edit files. Do not run destructive commands.
Return a concise Markdown report with findings, risks, concrete file references, suggested tests, and a safe branch plan.
Any command that would mutate repo state must be written as a proposed command and clearly marked as requiring a manual dry-run first.

Repository: /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main
Current branch: feature/mcp-director
Task id: task_20260508T143224Z_d3a621be
Task title: DSP Agent: repository task
Agent focus: audio DSP, mix pipeline math, LUFS/true-peak behavior, routing, WING/live-safety constraints

User task:
Analyze EQ decision logic in AUTO-MIXER-Tubeslave and create a dry-run report. Do not modify files.'
```

## Safety

- The command uses Codex read-only sandbox.
- Apply any generated implementation in a separate feature branch only after manual review.
- Do not run destructive git or filesystem commands unless they are first reviewed as dry-run proposals.
