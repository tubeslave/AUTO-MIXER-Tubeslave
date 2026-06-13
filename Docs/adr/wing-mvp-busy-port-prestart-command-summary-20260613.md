# ADR: WING MVP Busy-Port Prestart Command Summary

Date: 2026-06-13
Status: Accepted

## Context

The supervised WING Rack MVP still waits for real WING read-only connection
proof. The no-WING backend-port prestart gate can fail closed on an existing
local listener before any real preflight begins. The handoff and proof-plan
checks already require `backend_port_prestart_check_command`, but the first
failed busy-port summary did not carry the same command field.

## Options considered

1. Keep the exact prestart command only in handoff/proof-plan summaries.
2. Also include the exact prestart command in failed busy-port no-WING summaries.

## Decision

Use option 2. The failed `backend_port_free` detail now includes
`backend_port_prestart_check_command`, and the Markdown summary renders it for
operator review.

## Why this won

The first fail-closed artifact now carries the same command contract as
downstream handoff and proof-plan checks. This reduces ambiguity before a real
WING read-only preflight and keeps the operator-facing evidence self-contained.

## Rejected alternatives

Leaving the command only in later handoff summaries would preserve current test
coverage, but it would make the failed prestart artifact less actionable for
manual review.

## Implementation plan

- Add the command to busy-port `backend_port_free` detail.
- Render the command in no-WING Markdown summaries.
- Update WING MVP rails, autoloop docs, and generated latest progress.
- Record a validated report in the first-blocker evidence list.

## Test plan

- Python compile check for `tools/no_wing_release_check.py`.
- Focused no-WING/WING MVP rail tests.
- no-WING quick gate with expected fail-closed busy-port summary.
- Project Director, direction guard, memory lint, and full pytest.

## Risks and rollback

Risk is limited to stricter summary output. Rollback is to remove the command
from `backend_port_free` detail and Markdown rendering, but that would restore
the split between the first failed prestart artifact and downstream command
contract checks.
