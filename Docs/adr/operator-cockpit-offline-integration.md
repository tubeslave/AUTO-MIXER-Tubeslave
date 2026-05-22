# Operator Cockpit Offline Integration

## Context

The desired product UI is a cockpit with operator modes, a Director, specialized
agents, channel inventory, connection topology, and an AI proposal queue. Wing
Rack is not connected, so integration must stay offline-safe.

## Options considered

- Keep the frontend as static mock screens.
- Start a real multi-agent runtime and let it drive the UI.
- Add read-only product contracts and connect the UI to those contracts.

## Decision

Add a backend `operator_agent_roster` contract and mount the cockpit screens
against backend product-state endpoints. Do not start live automation and do not
unlock Auto mode.

## Why this won

This makes frontend and backend converge on the product architecture while
preserving the live-sound safety boundary. The UI can show Director + agents and
proposal queue truthfully even while no mixer is connected.

## Rejected alternatives

- Static mock agents were rejected because they would make the UI look more
  ready than the backend.
- Starting the real runtime was rejected because it increases live-control risk
  before the supervised path is fully proven.

## Implementation plan

- Build agent roster from current operator mode, audio capture, connection, and
  proposal queue state.
- Add WebSocket handler `get_operator_agent_roster`.
- Connect `App.js` to mode, dashboard, channel inventory, decision queue, roster,
  and topology messages.
- Route proposal UI actions through existing proposal queue handlers.
- Keep Console Executor blocked unless supervised mode and mixer connection are
  present.

## Test plan

- Backend product-state pytest.
- Frontend WebSocket transport tests.
- Frontend production build.
- Local rendered smoke: dashboard visible, mode switch works, agents roster
  visible, executor blocked offline.

## Risks and rollback

Risk: UI can still be misunderstood as live-ready. Mitigation: executor status
and mode policy expose blocked reasons directly. Rollback by removing the new
roster endpoint and returning `App.js` to the previous tab routing.
