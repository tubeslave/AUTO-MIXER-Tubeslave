# Brief: WING MVP no-WING prestart summary contract

Task id: `wing-mvp-nowing-prestart-summary-contract-20260613`

## Context

The supervised WING Rack MVP is still waiting on real WING read-only
connection proof. Before the intended backend/WING session, the operator must
run the no-WING quick gate and use its passed `summary.json` as the prestart
trace for the read-only preflight plan.

## Problem

The runner, evidence intake, review ledger, and proof plan now use shared
prestart contract constants, but `tools/no_wing_release_check.py` still had
local string checks for the no-WING summary type, connection policy, backend
check name, and passed statuses. That release gate is the source of the
prestart summary, so drift there would weaken the first MVP blocker path.

## Goal

Bind no-WING release-gate prestart summary generation and validation to the
same `backend/wing_readonly_preflight_contract.py` constants used by the
read-only preflight path.

## Non-goals

- No WING hardware contact.
- No backend start/stop.
- No readiness closure.
- No supervised proof execution.

## Validation

- Focused no-WING, runner, evidence, review, and proof-plan tests.
- no-WING quick release check.
- Direction guard and Project Director.
- Full pytest before commit.
