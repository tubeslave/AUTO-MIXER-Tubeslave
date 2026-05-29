# Codex Cycle

## Purpose

Codex should use this cycle before every new Automixer task.

## Cycle

READ MEMORY
-> CHECK PROJECT GOAL
-> CHECK SAFETY POLICY
-> INSPECT CURRENT TASKS
-> PLAN
-> EXECUTE SAFE CHANGES
-> TEST
-> LOG DECISION
-> WRITE SESSION REPORT
-> RUN MEMORY LINT
-> EXPLAIN RESULT SIMPLY

## Required Reads

- `AGENTS.md`
- `CLAUDE.md`
- `.ai/agent-core/rules.md`
- `.ai/agent-core/safety.md`
- `.ai/project-layer/automixer_goal.md`
- `.ai/project-layer/live_safety_rules.md`
- `.ai/project-layer/current_tasks.md`

## Execution Rules

- Do not ask for confirmation for safe documentation, tests, and dry-run
  utility work when the task is clear.
- Do not perform live OSC/WING writes.
- Do not alter production mixer logic unless the task requires it.
- Preserve unrelated local changes.
- Use focused tests for the touched area.
- Report full-test failures separately from focused-test results.

## Output

End with:

- what changed;
- tests run;
- memory lint result;
- safety status;
- risks;
- next steps;
- simple human explanation.
