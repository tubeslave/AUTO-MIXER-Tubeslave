# Operator Cockpit Offline Integration - Codex Proposal

## Short thesis

Use the existing Product Layer as the contract between frontend and backend,
then add a read-only agent roster so the cockpit reflects Director + specialized
agents without starting live automation.

## What I understood

The frontend concept is directionally correct: modes, agent console, channel
inventory, proposal queue, and supervised apply are the product shape. The
backend must move toward that shape without touching a real Wing Rack.

## Proposed solution

- Add a backend `operator_agent_roster` product endpoint.
- Mount the cockpit dashboard, agents, channels, and system views in `App.js`.
- Refresh mode, dashboard, channel inventory, queue, roster, and topology over
  WebSocket.
- Route proposal buttons through backend proposal handlers.
- Keep console execution blocked offline.

## Likely files to touch

- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `frontend/src/App.js`
- `frontend/src/components/ControlCenterViews.js`
- `frontend/src/services/websocket.js`
- tests for backend product state and frontend transport

## Alternatives considered

- Reuse static frontend mock agents: rejected because it hides backend truth.
- Start the real mixing agent runtime: rejected because the current goal is
  offline-safe integration, not live automation.

## Risks

- UI may imply live readiness if status labels are too optimistic.
- Auto mode could be misunderstood as enabled if not explicitly locked by policy.

## Test plan

- `pytest tests/test_operator_product_state.py`
- frontend node tests
- frontend build
- rendered smoke: Dashboard loads, Mode changes, Agents shows roster, Executor
  remains blocked without mixer.

## Where the other agent may disagree

Kimi may prefer a deeper runtime agent manager before exposing the roster. My
recommendation is to expose the read-only contract first so frontend/backend
converge without increasing live risk.
