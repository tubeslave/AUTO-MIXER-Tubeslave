# Paperclip Director Workflow

This runbook defines the safe operating contract between Director/Custom GPT, Paperclip, Codex, the GPT control bridge, and Automixer execution agents.

It exists to prevent two failure modes:

- generic tasks with weak context;
- unsafe escalation from repo analysis into live WING/OSC control.

## Control Chain

Use this flow:

```text
User or operator request
-> Director or Custom GPT
-> read knowledge_context first
-> formulate or refine a TUB issue
-> assign execution mode
-> Paperclip issue
-> Codex agent execution
-> tests or review evidence
-> structured report back to Paperclip
-> Director or GPT summary to user
```

The watchdog is not the Director. It may preview or safely assign an existing issue, but it must not invent work, bypass Paperclip, or jump directly to WING/OSC execution.

## Mandatory Knowledge Context

`knowledge_context` is mandatory input for Director-style task creation and task reformulation.

Director or Custom GPT must:

1. Read `knowledge_context` before drafting the task or giving execution advice.
2. Prefer the supplied context over generic prior assumptions.
3. Quote or summarize only the minimal relevant parts.
4. Refuse to produce a confident execution plan if `knowledge_context` is missing, empty, obviously stale, or unrelated.
5. Ask for a refreshed search or bridge context instead of filling gaps with generic boilerplate.

Minimum expected `knowledge_context` payload:

```yaml
knowledge_context:
  source: http://127.0.0.1:8788/v1/knowledge/automixer/search
  query: <issue title + objective + safety-sensitive keywords>
  top_hits:
    - <file path or paperclip object>
    - <short relevant excerpt>
```

Recommended supporting context:

- `GET /v1/context/automixer` for Paperclip and watchdog state;
- relevant repo docs or prior TUB reports;
- active safety constraints from the current issue.

If the request is ambiguous, Director should search first and only then formulate the task.

## Execution Modes

Every new TUB task must declare exactly one execution mode.

### 1. `dry_run_read_only`

Use for:

- repo analysis;
- audits;
- design/docs/planning;
- preview flows;
- Paperclip triage;
- report generation.

Allowed:

- read-only bridge usage;
- repo inspection;
- tests that do not touch live console state;
- Paperclip comments, issue docs, and safe task coordination.

Forbidden:

- live WING writes;
- live OSC writes;
- direct wakeup bypasses;
- changing runtime mixer state.

This is the default mode unless the task explicitly requires something else.

### 2. `supervised_approval`

Use for:

- controlled dispatch preparation;
- dry-run to real-run promotion checks;
- supervised write rehearsals with clear approval gates;
- any task that prepares a possible console mutation later.

Required gates:

- named approval step;
- clear operator stop point;
- rollback or panic-stop path;
- documented safety limits;
- explicit statement of what remains dry-run vs what may write after approval.

Still forbidden without explicit approval:

- direct live WING/OSC execution.

### 3. `live_wing_execution`

Use only for tasks that are intentionally about real console control.

Required gates:

- explicit Dmitry confirmation for this specific action;
- named console target such as WING Rack;
- dry-run or supervised evidence showing readiness;
- parameter limits, rate limits, cooldowns, and panic-stop path;
- readback or verification plan;
- rollback path.

Hard rule:

- live write, WING runtime mutation, or OSC control is forbidden without separate explicit confirmation from Dmitry.

If the issue text does not contain that approval, Director must downgrade the task to `dry_run_read_only` or `supervised_approval`.

## TUB Task Template

Director should create or rewrite Paperclip tasks in this structure:

```md
Goal:
<single concrete outcome>

Execution mode:
<dry_run_read_only | supervised_approval | live_wing_execution>

Context:
- <what changed or why this matters now>
- <constraints from current architecture or issue chain>

knowledge_context:
source: <bridge endpoint or other source>
query: <search query used>
top_hits:
1. <path or object> - <why relevant>
2. <path or object> - <why relevant>

Relevant files:
- <repo path>
- <repo path>

Safety gates:
- <what is forbidden>
- <required approval or stop point>
- <limits, rate limits, cooldowns, panic stop, rollback, readback>

Tests or verification:
- <pytest target, script, doc check, preview command, or "docs-only">

Done criteria:
- <observable evidence of completion>
- <what must be linked back into Paperclip>
```

## Director Rules

Director or Custom GPT must:

- formulate the task only after reading `knowledge_context`;
- keep the issue narrow enough for one agent heartbeat or one reviewable slice;
- name concrete files instead of generic areas where possible;
- preserve the existing read-only and supervised-write boundaries;
- default to the safest mode that still moves the issue forward.

Director or Custom GPT must not:

- propose live WING/OSC actions as the default next step;
- collapse approval and execution into one instruction;
- route around Paperclip by telling Codex to act outside the issue system;
- duplicate large context blocks into every follow-up if a concise reference is enough.

## Telegram `/ask` And `/debate` Context Budget

`/ask` and `/debate` should pass enough context to be specific, but should not flood the model with repeated copies of the same material.

Use this policy:

1. Include the issue goal, execution mode, and top safety constraints once.
2. Include a compact `knowledge_context` block with top hits, not full raw documents.
3. Prefer file paths and short excerpts over repeated long quotations.
4. If the same bridge context was already sent in the current thread, send only the delta or refreshed hits.
5. If the payload grows too large, drop duplicated narrative before dropping safety constraints.

Priority order for retained context:

1. explicit live-safety rules;
2. current issue goal and done criteria;
3. top relevant files and reports;
4. optional background narrative.

## Repo Anchors

This workflow should be read together with:

- `tools/gpt_control_bridge/README.md`
- `tools/automixer_operator/README.md`
- `tools/automixer_operator/IPHONE_CODEX_RUNBOOK.md`
- `tools/paperclip_watchdog/README.md`
- `.paperclip/reports/tub351_watchdog_dispatch_architecture.md`
- `Docs/reports/tub/TUB-343-WING-Write-Gate-Report.md`

## Safe Example

```md
Goal:
Document the Director workflow for knowledge_context-aware Paperclip issue creation.

Execution mode:
dry_run_read_only

Context:
- GPT bridge and Telegram paths can now pass knowledge_context.
- We need a stable protocol before expanding agent usage.

knowledge_context:
source: http://127.0.0.1:8788/v1/knowledge/automixer/search
query: Paperclip Director workflow knowledge_context safety
top_hits:
1. tools/gpt_control_bridge/README.md - defines the read-only bridge boundary
2. tools/automixer_operator/README.md - defines operator preview and approval stops

Relevant files:
- Docs/PAPERCLIP_DIRECTOR_WORKFLOW.md
- tools/gpt_control_bridge/README.md
- tools/automixer_operator/README.md

Safety gates:
- No WING/OSC/live runtime writes
- No Paperclip bypass
- No direct watchdog dispatch without explicit approval

Tests or verification:
- docs-only review
- verify linked files and command examples

Done criteria:
- checked-in workflow doc exists
- bridge/operator docs link to it
- issue comment links the resulting files
```
