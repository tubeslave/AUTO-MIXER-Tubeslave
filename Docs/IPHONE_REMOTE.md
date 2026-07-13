# iPhone / Remote Browser Access

> Code-side remote session authentication exists, but remote operation is not
> release-ready. The backend remains loopback-only. The only supported target
> topology is a trusted same-origin HTTPS/WSS reverse proxy to
> `127.0.0.1:8765`. Token provisioning/rotation, production proxy configuration
> and physical iPhone validation are still unproven. Authentication never grants
> mixer write authority.

## Current truth

- The active Vite entrypoint mounts the main Automixer cockpit. The historical
  `IPhoneControlSurface` component is not an active product route.
- Remote pages served over plain HTTP are rejected before a WebSocket is
  constructed.
- Backend startup also rejects every configured non-loopback `http://` Origin;
  only loopback development Origins may use HTTP.
- The backend refuses non-loopback binds even when a token is configured.
- Direct LAN access to port `8765`, `ws://<macbook-ip>:8765`, and the old
  `http://<macbook-ip>:3000` preview are unsupported.
- The browser auth gate accepts a session token in memory only. The repository
  does not yet contain a production pairing, secure delivery, rotation, or
  recovery workflow.
- Bulk `bypass_mixer`, trim reset, EQ reset and phase reset commands are
  fail-closed because they have not been migrated to the supervised
  readback/rollback executor.

This file is therefore an architecture and acceptance contract, not an
operator runbook.

## Supported target topology

```text
iPhone Safari / Home Screen app
        HTTPS + same-origin WSS /ws
                    |
          trusted TLS reverse proxy
                    |
        ws://127.0.0.1:8765 on MacBook
                    |
       authenticated read-only/proposal runtime
```

The reverse proxy must:

- serve the frontend and `/ws` from one trusted HTTPS origin;
- preserve the exact browser `Origin` header;
- terminate TLS with a certificate trusted by the physical iPhone;
- forward `/ws` only to `127.0.0.1:8765`;
- never publish backend port `8765` to the LAN;
- avoid putting the bearer token in URLs, cookies, proxy logs, or static files.

No production reverse-proxy configuration is currently accepted in this
repository, so this topology is not yet deployable from project instructions.

## Session authentication contract

- Token: exactly 43 base64url characters, representing 32 random bytes.
- Client ID: `automixer-ui`.
- Lifetime: from one minute through 24 hours; non-finite, zero, negative and
  longer values are rejected.
- Transport: token is sent only in the first WebSocket authentication frame.
- Storage: no token or endpoint persistence in local storage, session storage,
  URL parameters, fragments, or application logs.
- Origin: a remote browser may use only `wss://<current-https-origin>/ws`.
  Cross-origin, userinfo, query, fragment, plain WS, and remote-page-to-loopback
  targets are rejected before socket construction.
- Expiry: an active session is closed with code `1008` and returns to the auth
  gate. A replacement startup token currently requires a controlled backend
  restart; runtime token rotation is not implemented.

Authentication establishes a transport session only. It does not enable WING
writes, autonomous mode, bulk maintenance actions, or release readiness.

## Evidence required before this becomes a runbook

1. A reviewed production static-hosting and reverse-proxy configuration.
2. A trusted certificate chain verified on the exact physical iPhone.
3. An operator-accessible, non-disclosing token issuance and controlled restart
   or rotation procedure.
4. Proof that plain HTTP, direct port access, wrong Origin, wrong token, expired
   token, cross-origin WSS and remote-to-loopback token forwarding all fail
   before application commands can run.
5. Proof that the token is absent from browser storage, URLs, frontend/backend
   logs and proxy access logs.
6. Physical iPhone recovery proof for invalid token, expiry, reload, Home Screen
   relaunch and network loss.
7. Current-tree packaged-runtime proof under the immutable offline lock, with no
   WING contact and no live authority.

Until all seven exist, use only local Electron or local development access and
do not expose the backend to the LAN.
