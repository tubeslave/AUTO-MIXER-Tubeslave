# Brief: WING read-only plan stale prestart required condition

## Context

The read-only preflight runner and backend evidence contract require
`stale_backend_port_prestart_detail_contract` in offline plan fail-closed
conditions. The no-WING plan artifact validator's required list omitted that
marker, causing the generated quick-gate plan to fail its own release check.

## Task

Align the no-WING required fail-closed condition list with the runner and
backend evidence contract.

## Acceptance

- no-WING plan artifact validation accepts the generated fail-closed list.
- Tests assert the stale prestart detail marker remains required.
- No WING/backend/OSC contact is introduced.
