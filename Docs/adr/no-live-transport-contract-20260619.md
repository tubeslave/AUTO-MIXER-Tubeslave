# ADR: No-live-transport contract

## Context

Integration promotion still has a loose `no_live_transport` check. Top-level
integration bundle fields deny live writes and hardware labels, but a future
bundle needs explicit evidence that no WING/backend transport was opened, no
live write was sent, and only offline artifacts were produced.

## Options Considered

1. Keep `no_live_transport` as a loose boolean integration check.
2. Add a focused no-live-transport contract before binding integration bundles
   to it.
3. Treat source-code import checks as sufficient proof of no live transport.

## Decision

Use option 2. Add `backend/no_live_transport_contract.py` as a fail-closed
offline contract for no-live-transport evidence.

## Why This Won

The focused contract makes transport absence auditable. It requires controls,
evidence artifacts, zero runtime observations and no forbidden transport tokens
instead of trusting a single boolean. This is the right evidence shape before
future integration-bundle binding.

## Rejected Alternatives

Loose booleans were rejected because they can be set without runtime evidence.
Source-code import checks were rejected as the only proof because they do not
show what happened during a candidate run; they remain useful as regression
tests.

## Implementation Plan

1. Add `evaluate_no_live_transport_contract`.
2. Require offline executor, artifact-only output, no WING/OSC/websocket/MIDI
   transport, no backend/hardware contact, no live write requests and audit log
   controls.
3. Require source-audit, runtime-config, network-trace, artifact-manifest,
   operator-report and audit-reference evidence.
4. Require zero runtime observations for opened transport, live write requests,
   mixer-client calls and network connections.
5. The contract always returns `live_authority_granted: false` and
   `readiness_closure_allowed: false`.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_no_live_transport_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m022_no_live_transport_contract_stays_offline_and_artifact_only -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: future integration fixtures that only set `no_live_transport: true` will
remain incomplete once this contract is bound into integration promotion. This
is intentional: they need artifact-backed transport evidence.

Rollback: revert the M022 commit. This removes the standalone transport
contract without touching live transport or runtime mixer behavior.
