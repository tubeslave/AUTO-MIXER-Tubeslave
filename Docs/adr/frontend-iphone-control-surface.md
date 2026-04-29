# Frontend iPhone Control Surface

## Context

The operator needs a focused iPhone/iPad frontend for controlling AUTO-MIXER from the
same local network as the MacBook. The existing React UI is desktop/tab oriented and the
WebSocket client defaulted to `ws://localhost:8765`, which fails on iPhone because
`localhost` points to the phone, not the MacBook.

## Options considered

1. Add a dedicated mobile remote surface inside the existing React app.
2. Build a separate iPhone-only app or native iOS wrapper.
3. Only restyle the existing desktop tabs responsively.

## Decision

Add a dedicated `Remote` screen to the existing React app and serve it from the MacBook.
When the app is opened from `http://<macbook-ip>:3000`, the WebSocket URL is inferred as
`ws://<macbook-ip>:8765`. The backend entry point binds to `0.0.0.0` by default for local
network access.

## Why this won

It reuses existing WebSocket handlers and avoids introducing signing, App Store, or
native iOS dependencies. It also keeps the phone UI focused on safe operation: status,
channel selection, pending corrections, backup, rollback, apply, bypass, and emergency
stop.

## Rejected alternatives

- Native iOS app: better device integration, but much higher setup and signing cost.
- Desktop tabs only: lower implementation cost, but poor touch ergonomics and too much
  information density during live operation.

## Implementation plan

- Add `IPhoneControlSurface` and mobile CSS.
- Add `Remote` navigation and auto-open it on narrow screens.
- Infer WebSocket URL from the page host, with manual override in Setup.
- Add frontend service helpers for backup/restore, mixer scan, audio scan, and backend
  URL updates.
- Add iPhone Home Screen metadata and manifest.
- Document MacBook/iPhone setup in `Docs/IPHONE_REMOTE.md`.

## Test plan

- `cd frontend && npm test`
- `cd frontend && npm run build`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- Manual iPhone test on the same Wi-Fi: open `http://<macbook-ip>:3000`, connect backend,
  select audio device, select channels, create backup, review/apply pending actions,
  restore backup, and use emergency stop.

## Risks and rollback

- Local network access depends on macOS Firewall permissions for ports `3000` and `8765`.
- `Bypass` is intentionally dangerous and remains behind confirmation.
- If mobile access is not desired, start backend with `AUTOMIXER_WS_HOST=localhost`.
- Roll back by removing the `Remote` tab/component and restoring the previous WebSocket
  default URL behavior.
