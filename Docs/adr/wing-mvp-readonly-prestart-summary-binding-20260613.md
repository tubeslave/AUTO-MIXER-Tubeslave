# ADR: WING MVP Read-Only Prestart Summary Binding

## Context

The WING read-only preflight path already requires a backend-port prestart
contract. Before this change, the read-only artifact could prove the contract
field names but did not prove which no-WING `summary.json` actually passed.

## Options considered

1. Keep declarative prestart fields only.
2. Require a no-WING prestart summary path and SHA256 in the read-only artifact.
3. Store the prestart hash only in the operator decision JSON.
4. Treat the handoff packet as sufficient prestart evidence.

## Decision

Use option 2.

## Why this won

The prestart check must happen before backend/WING WebSocket use. Binding the
prestart artifact in the runner blocks stale or missing prestart evidence before
the first read-only request is sent.

## Rejected alternatives

- Option 1 can accept a candidate with no concrete prestart artifact.
- Option 3 catches the issue too late, after candidate capture.
- Option 4 would confuse offline guidance with proof.

## Implementation plan

- Add `--prestart-summary` validation to the runner.
- Persist prestart path, SHA256, status, policy, live-write state, and
  `backend_port_free` status in plan and summary artifacts.
- Validate the referenced prestart file in evidence intake.
- Require the same binding in decision-ledger and proof-plan readiness checks.
- Update runbook and MVP rails.

## Test plan

- Runner tests for valid and missing prestart summaries.
- Evidence tests for missing and mismatched prestart summary files.
- Review, decision, proof-plan, Project Director, and rail tests.
- Full no-WING/MVP focused validation.

## Risks and rollback

Rollback removes the new CLI flag and prestart-summary fields. The prior
read-only preflight path would still be read-only, but would again rely on a
weaker declaration-only prestart trace.
