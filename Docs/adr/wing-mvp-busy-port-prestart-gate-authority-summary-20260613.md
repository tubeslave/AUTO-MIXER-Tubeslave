# ADR: WING MVP Busy-Port Prestart Gate Authority Summary

## Context

The current WING MVP first blocker is `WING read-only connection proof`. The
backend-port prestart gate can fail closed before any intended backend/WING
session when a local listener is already present.

The failed busy-port summary already exposes the prestart command and
required/must-pass flags. It should also explicitly show that the failed gate
does not authorize execution, readiness closure, or live console authority.

## Options Considered

1. Add only documentation wording.
2. Add generic `live_authority_granted: false` to the failed summary.
3. Add prestart-gate-scoped authority-denial fields to JSON, Markdown, contract,
   and rails.

## Decision

Use option 3.

## Why This Won

The prestart artifact is the operator's first fail-closed evidence before a real
WING session. Making authority denial explicit in that artifact reduces the
chance that a failed no-WING summary or handoff packet is mistaken for permission
to proceed.

## Rejected Alternatives

Documentation-only wording was too weak for machine checks. A single generic
live-authority field did not cover execution or readiness closure.

## Implementation Plan

- Add prestart-gate authority-denial fields to the shared failure-detail
  contract.
- Populate and render the fields for failed busy-port summaries.
- Require the fields in WING MVP docs and autoloop rails.
- Add tests covering JSON detail, Markdown output, Project Director, and WING
  rails.

## Test Plan

- `py_compile` on touched Python modules.
- Focused no-WING and WING MVP rails tests.
- no-WING quick expected fail-closed busy-port run.
- Project Director and direction guard checks.
- Full pytest.

## Risks And Rollback

The main risk is stale downstream docs/tests because the failure-detail field
list is exact. Roll back by removing the three prestart-gate fields from the
contract, summary detail, docs, and tests.
