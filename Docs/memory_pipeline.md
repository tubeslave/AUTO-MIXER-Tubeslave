# Automixer Memory Pipeline

## Purpose

Automixer needs long-term Markdown memory because agents work across many
cycles: Codex, Kimi, Paperclip, reports, ADRs, tests, and live-safety notes all
need one stable place to agree on current project intent.

The memory pipeline does not control the mixer. It documents rules, records
decisions, captures experiments, and helps agents start each task with the same
safety assumptions.

## Structure

The memory layer has three parts:

- `.ai/agent-core/`: shared agent rules, workflow, safety, source policy, and
  lint policy.
- `.ai/project-layer/`: Automixer-specific goal, architecture, WING/OSC policy,
  gain-staging method, safety rules, tasks, decisions, experiments, reports, and
  glossary.
- `.ai/pipeline/`: CAPTURE -> CLASSIFY -> PROPOSE -> APPROVE -> APPLY -> LOG ->
  LINT, plus Codex and Paperclip cycles and templates.

Existing Council artifacts still matter:

- `.ai/briefs/`
- `.ai/proposals/`
- `.ai/reviews/`
- `.ai/memory/project.md`
- `Docs/adr/`
- `.paperclip/reports/`

## How Codex Should Read Memory

Before work, Codex should read:

1. `AGENTS.md`
2. `CLAUDE.md`
3. `.ai/agent-core/rules.md`
4. `.ai/agent-core/safety.md`
5. `.ai/project-layer/automixer_goal.md`
6. `.ai/project-layer/live_safety_rules.md`
7. `.ai/project-layer/current_tasks.md`

For WING, OSC, Dante, gain, trim, fader, or live safety tasks, Codex should also
read:

- `.ai/project-layer/wing_osc_policy.md`
- `Docs/TOOLS.md`
- `Docs/CONVENTIONS.md`
- relevant ADRs and recent reports

## How Paperclip Should Write Reports

Paperclip-facing automation should write human-readable reports into
`.paperclip/reports/`. A useful report states:

- task mode: audit-only, dry-run, supervised-write, or live-write;
- files inspected or changed;
- tests and checks run;
- whether live OSC/WING writes occurred;
- risks and next steps.

Paperclip must not turn an issue, report, or recommendation into live console
control without explicit approval.

## How Decisions Enter decisions.md

Small project-layer decisions can be appended with:

```bash
python3 tools/memory/append_decision.py \
  --title "Decision title" \
  --decision "What was decided" \
  --context "Why this came up" \
  --tests "Relevant test or doc check"
```

Architecture-significant decisions still belong in `Docs/adr/`. The
project-layer `decisions.md` file is a working memory index, not a replacement
for ADRs.

## How Soundcheck Experiments Are Recorded

Raw observations can be appended with:

```bash
python3 tools/memory/capture_note.py \
  --title "Soundcheck observation" \
  --mode "dry-run" \
  --source "report or command path" \
  --note "What was observed"
```

Use `audit-only` or `dry-run` unless there was explicit supervised approval for
the exact action. Offline or dry-run evidence must not be described as live
readiness.

## How Dangerous Console Control Is Prevented

The memory layer repeats the same hard boundary in multiple places:

- live OSC writes are forbidden without explicit approval;
- live WING writes are forbidden without explicit approval;
- WING, OSC, Dante, gain, trim, fader, routing, mute, snapshot, EQ, dynamics,
  and FX actions default to audit-only or dry-run;
- supervised writes require pre-read, safety check, explicit approval, readback,
  and rollback;
- recommendations are not control commands.

The memory scripts do not import or call WING, OSC, Dante, audio capture,
backend handlers, or live runtime clients.

## Commands

Run memory lint:

```bash
python3 tools/memory/memory_lint.py
```

Run focused memory tests:

```bash
python3 -m pytest -q tests/test_memory_pipeline.py
```

Run the standard project test gate when practical:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

The repository does not have a root Makefile. Use the Python commands above
instead of adding a new task runner only for memory docs.
