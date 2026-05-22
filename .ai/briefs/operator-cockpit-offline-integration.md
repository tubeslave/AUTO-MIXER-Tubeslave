# Operator Cockpit Offline Integration

## Task

Connect the frontend cockpit screens to backend product contracts without a real
Wing Rack connection.

## Goal

- Show backend operator mode in the UI.
- Show the backend proposal queue in the Director/Agents screens.
- Show a truthful multi-agent roster where Director coordinates specialized
  agents.
- Keep live execution blocked unless supervised mode and mixer connection are
  both present.

## Constraints

- No real mixer connection.
- No live WING writes.
- No autonomous Auto-mode enablement.
- Direct console executor remains blocked offline.

## Verification

- Backend product-state tests.
- Frontend transport tests and build.
- Rendered local UI smoke via headless Chrome.
