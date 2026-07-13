# Secure remote browser access

## Context

The historical iPhone design used a LAN HTTP page and a direct
`ws://<macbook-ip>:8765` backend. That exposes bearer traffic without TLS,
conflicts with the backend's loopback-only bind, and describes a mobile route
that is not mounted by the active Vite entrypoint. The current code has an
in-memory authentication gate, exact Origin parsing, token expiry and a
loopback-only backend, but it does not yet provide production proxy deployment,
token provisioning/rotation, or physical-device evidence.

## Options considered

1. Re-enable a token-authenticated backend bind on the LAN.
2. Keep the backend loopback-only and terminate trusted HTTPS/WSS at a
   same-origin reverse proxy.
3. Ship a native iOS application with its own secure transport.
4. Keep remote access unsupported and use local Electron only.

## Decision

The only supported target architecture is a trusted same-origin HTTPS/WSS
reverse proxy that forwards `/ws` to `127.0.0.1:8765`. The backend must never
bind directly to a non-loopback interface. A remote browser may connect only to
`wss://<current-page-origin>/ws`, using an exact 43-character base64url startup
token and `client_id=automixer-ui` in the first frame. Tokens remain memory-only
and expire within 24 hours. Authentication does not grant mixer write authority.

This decision defines a target, not a release-ready deployment. Remote access
remains blocked until proxy, certificate, provisioning and physical iPhone
evidence are accepted.

## Why this won

It keeps the live backend off the LAN, gives the browser a trusted TLS origin,
prevents a remote page from redirecting its token to a phone-local service, and
preserves the existing fail-closed operator authority model without adding a
native-app dependency.

## Rejected alternatives

- Direct LAN backend bind: rejected because TLS, Origin enforcement and token
  disclosure would depend on the backend port itself being safely exposed.
- Native iOS application: deferred because it adds signing and distribution
  work without removing the need for backend transport and authority gates.
- Treating the historical HTTP preview as a runbook: rejected because current
  code blocks it and the described mobile route is not active.
- Claiming remote readiness from unit tests alone: rejected because trusted TLS,
  proxy behavior and device recovery require external evidence.

## Implementation plan

- Keep `AutoMixerServer` loopback-only and exact-Origin authenticated.
- Reject configured non-loopback HTTP Origins at backend startup; HTTP remains
  valid only for the loopback development UI.
- Keep browser endpoints/token memory-only and same-origin WSS-only.
- Add a reviewed reverse-proxy/static-hosting configuration without publishing
  port `8765`.
- Add an operator-controlled token issuance and rotation/restart procedure that
  never exposes plaintext in logs or URLs.
- Decide whether the active cockpit becomes responsive remote UI or a new mobile
  route is implemented; do not revive the dormant component implicitly.
- Keep all remote sessions under the same WING write/readback/rollback gates as
  local Electron.

## Test plan

- Unit tests for exact endpoint/origin/token/TTL/client-id policy.
- WebSocket tests for invalid and expired token close code `1008` and recovery.
- Browser tests proving no token persistence or URL/log disclosure.
- Reverse-proxy tests for trusted HTTPS/WSS, preserved Origin and closed direct
  backend port.
- Physical iPhone tests for invalid token, expiry, reload, relaunch and network
  loss, first under packaged offline lock and without WING contact.

## Risks and rollback

The main risks are certificate trust failure, proxy log disclosure, stale token
delivery, and an apparent remote connection being mistaken for write authority.
Rollback is to remove the proxy route and keep the backend on loopback; local
Electron remains the supported path. No rollback may widen the backend bind or
restore HTTP/WS LAN instructions.
