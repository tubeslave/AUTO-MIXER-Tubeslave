# Task automixer-dashboard-main: Automixer main Dashboard

## Problem

Build the main Automixer Dashboard frontend as a production-oriented React/TypeScript
surface in the dark high-tech AI live mixing console style. The page must be dense
enough for operator use and structured for future WebSocket/OSC telemetry without
connecting real OSC write commands.

## Constraints

- Frontend only; do not modify backend or add backend migrations.
- Use mock data for now and keep it in separate frontend mock files.
- Do not send live OSC writes or introduce any UI path that implies direct write access.
- Preserve the existing Product Layer/WebSocket direction and keep future integration
  points explicit with TODO comments.
- Preserve safety-critical language around safe limits, supervised gate, and rollback.
- Avoid new production dependencies.

## Definition of Done

- Dashboard route renders a dense desktop-first console view with topbar, sidebar,
  analytics cards, 14 channel strips, analysis cards, AI Director panel, and footer
  transport.
- Selected Lead Vocal channel is visually highlighted.
- Accept/reject/transport actions remain local UI mock actions only.
- Frontend build passes.

## Test Command

`npm --prefix frontend run build`
