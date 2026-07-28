# automixer-autopilot

Use this skill for scheduled Automixer autonomous development cycles.

## Mandatory read set

Read these first:

1. `AGENTS.md`
2. `CLAUDE.md`
3. `Docs/PROJECT_EXECUTION_CONTRACT.md`
4. `Docs/PROJECT_GOAL.md`
5. `Docs/PROJECT_COMPLETION_GATE.md`
6. `Docs/CURRENT_STATE.md`
7. `Docs/ACTIVE_MILESTONE.md`
8. active plan under `Docs/plans/`
9. `ops/autopilot/STATE.json`
10. `ops/autopilot/BLOCKERS.md`
11. recent git status and diff

Read targeted source files only after the active checkpoint identifies them.

## Wake-up procedure

1. If `ops/autopilot/STOP` exists, stop new work, release lease, set status to
   `STOPPED`, update heartbeat/report/ledger and exit.
2. Acquire `ops/autopilot/WRITER_LEASE/` atomically. If a non-stale writer owns
   it, exit without product edits.
3. Respect existing `.codex_autoloop.lock` if present and non-stale.
4. Resume from the first checkpoint without verified evidence.
5. Make the smallest scoped change required by the checkpoint.
6. Run targeted checks.
7. Record evidence in `RUN_LEDGER.jsonl`, `STATE.json`, `HEARTBEAT.md` and
   `LAST_REPORT.md`.
8. Do not mark a checkpoint complete while required checks fail.
9. At milestone boundary, run the milestone validation contract and independent
   safety/regression review.

## Truth and validation ordering

1. Write `STATE.json`, `HEARTBEAT.md`, `LAST_REPORT.md`, `RUN_LEDGER.jsonl`
   and every operator handoff surface first; run rails and the full suite
   after the final truth write. Validation that predates a truth edit is
   void.
2. New `RUN_LEDGER.jsonl` rows use the complete handoff schema: `authority`,
   `decision_ledger`, `current_blocker`, `next_operator_action`,
   `external_validation_requirements`, `hardware_contacted`, `live_writes`.
3. Never deselect failing rails to make a run pass. A red dynamic CP05 test
   on a dirty mid-iteration tree is resolved only by the source commit plus
   exactly one marker-only CP05 rebind.
4. Prefix any new full-suite result string in ledger `results`; the bare
   `N passed, M skipped` format is bound to the committed M115 baseline.
5. Before reporting, verify every declared deliverable is tracked by git.

## Safety

Never run real-console control during unattended automation. Never send live OSC
writes. Never widen WING live scope beyond supervised `fader`/`gain` without a
separate ADR and hardware proof.

## Model budget

Use economical reasoning for state checks, documentation sync and obvious
bounded edits. Use stronger reasoning only for architecture, difficult DSP,
cross-system debugging, safety-critical implementation and independent
milestone review. Prefer the repo-local budget policy in
`.ai/agent-core/model_budget_policy.md`.

## Output expectations

Human-facing reports must be in Russian. Technical identifiers may remain in
English. Always name exact files, commands, evidence paths, blockers and next
action.

