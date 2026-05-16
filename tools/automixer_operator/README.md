# Automixer Operator

Safe one-command operator workflow for Automixer automation.

The operator aggregates GPT control bridge context, Paperclip status, watchdog state, and the latest watchdog report. It does not expose real dispatch, wakeup, WING/OSC, GitHub, commit, push, or issue-status write actions.

## Codex On iPhone

Use Codex from ChatGPT on iPhone as the execution console, but keep Paperclip in the loop:

1. Ask Codex to run `bridge-start` if the bridge is not running.
2. Ask Codex to run `doctor`.
3. Ask Codex to run `status`.
4. Ask Codex to run `target` if you want to see the selected safe Paperclip issue/agent.
5. Ask Codex to run `next` to see the safe next step.
6. Ask Codex to run `cycle` when you want the full safe dry-run flow in one command.
7. Review `.paperclip/reports/watchdog_dispatch_report.md`.
8. Ask Codex to run `approve-template` only after the preview report is acceptable.
9. Manually approve any real dispatch command before it is run.
10. Ask Codex to run `monitor` after an approved dispatch.

Codex must not skip Paperclip or call WING/OSC/live runtime paths from this operator.

The detailed iPhone operating contract is in:

```text
tools/automixer_operator/IPHONE_CODEX_RUNBOOK.md
```

## Status

```bash
python3 tools/automixer_operator/operator.py status
```

Shows:

- GPT control bridge context source;
- Paperclip API health and company id;
- active runs;
- agents summary;
- blocked/todo/in_progress queue counters when Paperclip dashboard data is available;
- `.paperclip/watchdog_state.json` summary;
- latest watchdog report path;
- `recommended_next_action`.

If the bridge process on `127.0.0.1:8788` is not running, the operator falls back to the same local read-only bridge module and still handles missing Paperclip gracefully.

## Bridge Runtime

```bash
python3 tools/automixer_operator/operator.py bridge-status
python3 tools/automixer_operator/operator.py bridge-start
```

`bridge-start` starts only the read-only GPT control bridge on loopback through tmux. It refuses non-loopback hosts and does nothing if the bridge is already healthy.

## Doctor

```bash
python3 tools/automixer_operator/operator.py doctor
```

Checks the local operator loop:

- bridge health endpoint is reachable;
- bridge reports `read_only=true`;
- Paperclip is reachable through context;
- write routes are disabled;
- dispatch is blocked;
- WING/OSC is blocked.

## Preview

```bash
python3 tools/automixer_operator/operator.py preview
```

Runs the existing watchdog `dispatch-preview` command only:

```bash
python3 tools/paperclip_watchdog/automixer_paperclip_watchdog.py dispatch-preview
```

The operator forces these safety flags for the preview subprocess:

- `AUTOMIXER_WATCHDOG_DRY_RUN=true`
- `AUTOMIXER_WATCHDOG_DISPATCH_ENABLED=true`
- `AUTOMIXER_WATCHDOG_DISPATCH_DRY_RUN=true`
- `AUTOMIXER_WATCHDOG_DISPATCH_METHOD=patch_issue_assignment`
- `AUTOMIXER_WATCHDOG_DISPATCH_FORBID_DIRECT_WAKEUP_WITHOUT_ASSIGNMENT=true`
- `AUTOMIXER_WATCHDOG_DISPATCH_FORBID_LIVE_OSC=true`
- `AUTOMIXER_WATCHDOG_DISPATCH_FORBID_WING_RUNTIME_WRITE=true`

Preview may write the local watchdog report/state artifacts, but it must not send real Paperclip dispatch writes.

Before launching the dry-run preview, the operator tries to auto-select a safe Paperclip target:

- issue must be `todo`;
- issue must be unassigned;
- issue must not have an active run;
- agent must be idle/available/ready;
- agent must use `codex_local`.

If a target is found, its issue id, agent id, and task key are passed only to the dry-run watchdog subprocess.

Report path:

```text
.paperclip/reports/watchdog_dispatch_report.md
```

## Target

```bash
python3 tools/automixer_operator/operator.py target
```

Reads Paperclip through GET requests and prints the currently selected safe preview target. This command does not dispatch, does not change issue assignment, and does not wake an agent.

## Cycle

```bash
python3 tools/automixer_operator/operator.py cycle
```

Runs the complete safe operator pass:

```text
status/context -> watchdog dry-run preview -> refreshed recommendation -> latest dry-run report -> approval command
```

`cycle` still does not execute real dispatch. It only prints the command that would require explicit user approval.

## Next

```bash
python3 tools/automixer_operator/operator.py next
```

Prints the safe next operator action:

- check Paperclip if Paperclip is down;
- monitor if active runs exist;
- select/create a new unassigned Paperclip todo issue if no safe target exists;
- show approval template if the latest dry-run preview is ready;
- create/select a new Paperclip todo issue if the current report is blocked;
- otherwise run `cycle`.

## Approval Template

```bash
python3 tools/automixer_operator/operator.py approve-template
```

Reads the latest dry-run watchdog dispatch report, extracts:

- issue id;
- selected agent id;
- idempotency key.

Then it prints a ready-to-review real dispatch command. It does not execute that command.

The real command remains an explicit separate operator approval step through the existing watchdog:

```bash
python3 tools/paperclip_watchdog/automixer_paperclip_watchdog.py dispatch
```

## Monitor

```bash
python3 tools/automixer_operator/operator.py monitor
```

Reads active runs, the latest watchdog report, issue status, run status, and comments/report data where Paperclip exposes them. It does not change the issue, does not wake agents, and does not launch new work.

## Hard Blocks

This operator intentionally has no executable command for:

- real dispatch;
- agent wakeup;
- WING/OSC/live runtime;
- GitHub writes;
- commit/push;
- issue status changes;
- `POST`, `PATCH`, `DELETE`, or `PUT` routes.

The only future-approved write remains the separate approval command printed by `approve-template` or `preview`, and it must be run manually by the operator.

## Operator Responsibility

The operator must review the dry-run report before running any real dispatch command. In particular, check:

- issue id;
- selected agent;
- idempotency key;
- active runs;
- blocked safety checks;
- `dispatch_executed: False` in the preview report.
