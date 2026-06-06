# ADR: WING Supervised MVP Rails

## Context

Codex work has started moving toward WING proof, but some cycles still drift
into memory, queue-import, result-consumer, and report-only layers. The project
needs one explicit MVP rail that keeps the next task tied to a real supervised
WING Rack blocker.

## Options considered

1. Keep relying on the current prompt and reports.
2. Add a dedicated MVP contract, blocker table, runbooks, and progress report.
3. Install another background automation service.

## Decision

Use option 2.

## Why this won

The MVP contract and blocker table make the next task visible to Codex and to
the operator. Runbooks provide exact hardware steps. A progress report gives a
single current status without granting live authority.

## Rejected alternatives

- Prompt-only direction improved behavior but did not provide a durable task
  table.
- A new background service is unnecessary because the existing Codex autoloop
  can run the guard and progress tooling.

## Implementation plan

1. Add `Docs/MVP_WING_SUPERVISED.md`.
2. Add `.ai/project-layer/wing_mvp_blockers.md`.
3. Add WING preflight and supervised proof runbooks.
4. Add a progress updater for `Docs/reports/MVP_PROGRESS_LATEST.md`.
5. Update Codex prompt and direction guard to treat the MVP rails as the
   current task source.

## Test plan

- Validate required MVP docs and runbook content.
- Validate progress updater parsing.
- Validate Codex direction guard stop-the-line behavior.
- Run memory lint and markdown diff checks.

## Risks and rollback

Risk: the blocker table may be mistaken for proof. Mitigation: every row says
real artifacts are required and no row is closed by offline evidence. Rollback
is removing the MVP rail files and prompt references.
