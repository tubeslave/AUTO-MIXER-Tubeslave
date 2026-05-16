# iPhone Codex Automixer Runbook

This runbook is the operating contract for using ChatGPT on iPhone as the control surface for the local Automixer repo.

Codex is the execution console. Paperclip remains the work system. The operator CLI is the safety layer between them.

## Golden Path

Use this order by default:

```text
bridge-start -> doctor -> status -> target -> next -> cycle -> report review -> approve-template -> explicit user approval -> real watchdog dispatch -> monitor
```

No step after `approve-template` is automatic.

## User Phrases

When the user says:

```text
Start Automixer bridge
```

Codex may run:

```bash
python3 tools/automixer_operator/operator.py bridge-start
```

When the user says:

```text
Check Automixer bridge
```

Codex may run:

```bash
python3 tools/automixer_operator/operator.py bridge-status
```

When the user says:

```text
Check Automixer safety
```

Codex may run:

```bash
python3 tools/automixer_operator/operator.py doctor
```

When the user says:

```text
Check Automixer
```

Codex may run:

```bash
python3 tools/automixer_operator/operator.py status
```

When the user says:

```text
What should Automixer do next?
```

Codex may run:

```bash
python3 tools/automixer_operator/operator.py next
```

When the user says:

```text
Show Automixer target
```

Codex may run:

```bash
python3 tools/automixer_operator/operator.py target
```

When the user says:

```text
Run the full safe cycle
```

Codex may run:

```bash
python3 tools/automixer_operator/operator.py cycle
```

When the user says:

```text
Do a safe preview
```

Codex may run:

```bash
python3 tools/automixer_operator/operator.py preview
```

When the user says:

```text
Show approval command
```

Codex may run:

```bash
python3 tools/automixer_operator/operator.py approve-template
```

When the user says:

```text
Monitor Automixer
```

Codex may run:

```bash
python3 tools/automixer_operator/operator.py monitor
```

## Required Stop Points

Codex must stop and ask for explicit user approval before:

- real watchdog dispatch;
- running a printed approval command;
- any `POST`, `PATCH`, `DELETE`, or `PUT`;
- any WING/OSC/live runtime action;
- any GitHub write, commit, push, merge, or review action;
- staging `.paperclip/`, artifacts, render outputs, config secrets, or runtime logs.

Approval must be concrete. Examples:

```text
Approved: run real watchdog dispatch for issue <issue-id> with agent <agent-id>.
```

Not enough:

```text
ok
go
continue
```

## Bridge Runtime

The read-only GPT control bridge should listen only on loopback:

```text
http://127.0.0.1:8788
```

Health check:

```bash
curl -s http://127.0.0.1:8788/health
```

Expected:

```json
{
  "ok": true,
  "read_only": true,
  "service": "automixer-gpt-control-bridge"
}
```

If the bridge is down, Codex may start it in tmux:

```bash
python3 tools/automixer_operator/operator.py bridge-start
```

Codex must not bind the bridge to `0.0.0.0`, LAN IPs, or public interfaces.

## Paperclip Health

Paperclip must remain reachable locally:

```bash
curl -s http://127.0.0.1:3100/api/health
```

If Paperclip is down, Codex should report that the operator loop is blocked. It should not bypass Paperclip by directly changing repo/runtime state.

## Preview Rules

`preview` is allowed because it forces dry-run safety flags.

Before preview, Codex may use `target` selection. A safe target means:

- Paperclip is reachable;
- active runs are zero;
- issue is `todo`;
- issue is unassigned;
- issue has no active run;
- agent is idle/available/ready;
- agent adapter is `codex_local`.

It must not:

- send `PATCH`;
- dispatch work;
- wake agents;
- touch WING/OSC;
- change issue status.

It may update the local watchdog report/state artifacts:

```text
.paperclip/reports/watchdog_dispatch_report.md
.paperclip/watchdog_state.json
```

These artifacts are runtime evidence and should not be staged unless the user explicitly asks.

## Approval Command Rules

`approve-template` prints a command. It does not execute it.

Before a real dispatch, Codex must confirm:

- latest report is dry-run, not old real dispatch;
- `dispatch_executed: False`;
- target issue id is correct;
- selected agent id is correct;
- idempotency key is present;
- active runs are zero or policy allows proceeding;
- user explicitly approved the exact dispatch.

## Monitor Rules

`monitor` is read-only.

It may read:

- active runs;
- issue status;
- comments/report;
- final result.

It must not:

- change issue status;
- wake agents;
- start new agents;
- re-dispatch.

## Minimal iPhone Session

Use this sequence from ChatGPT on iPhone:

```text
Codex, start Automixer bridge if needed.
Codex, check Automixer safety.
Codex, check Automixer status only.
Codex, show Automixer target only.
Codex, tell me the next safe Automixer step.
Codex, run the full safe cycle, no dispatch.
Codex, show me the approval template, do not execute it.
Codex, monitor the Automixer run read-only.
```

If Codex sees a write action, it must stop and ask first.
