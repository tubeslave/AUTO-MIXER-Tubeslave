# AUTO-MIXER MCP Director

Local stdio MCP server for routing GPT/Codex work into Paperclip-style task and report artifacts.

## Tools

- `create_task`: writes `.paperclip/tasks/*.json`.
- `run_agent`: creates a task and writes a DSP/ML/Refactor/Analyzer report. By default it is dry-run and does not invoke Codex.
- `run_full_review`: creates per-agent tasks and a combined review report.
- `git_report`: writes a read-only git report to `.paperclip/reports/*.md`.
- `telegram_notify`: writes a notification report and only calls Telegram when `dryRun:false`.

## Safety Defaults

- External side effects default to `dryRun:true`.
- Codex CLI execution, when explicitly enabled with `dryRun:false`, uses `codex exec -s read-only`.
- If `codex` is not in `PATH`, `run_agent` writes a manual Markdown fallback task under `.paperclip/reports`.
- The server refuses to write Paperclip artifacts on `main`, `master`, `production`, `prod`, or `release`.
- Set `MCP_DIRECTOR_REQUIRED_BRANCH=feature/mcp-director` if you want to hard-pin writes to this branch.
- The server never uses shell interpolation for git/Codex/Telegram calls.

## Run Locally

```bash
cd /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main
node tools/mcp_director/server.js
```

## Connect To Codex

From any terminal:

```bash
codex mcp add --env MCP_DIRECTOR_REPO=/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main automixer-director -- node /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/tools/mcp_director/server.js
codex mcp list
codex mcp get automixer-director
```

Optional stricter branch guard:

```bash
codex mcp add --env MCP_DIRECTOR_REPO=/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main --env MCP_DIRECTOR_REQUIRED_BRANCH=feature/mcp-director automixer-director -- node /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/tools/mcp_director/server.js
```

## Verify

```bash
cd /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/tools/mcp_director
npm run smoke
```
