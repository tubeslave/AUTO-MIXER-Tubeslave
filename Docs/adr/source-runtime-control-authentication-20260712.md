# Source Runtime Control Authentication — 2026-07-12

## Context

Loopback binding and browser Origin checks do not authenticate native local
processes. A source backend started without `AUTOMIXER_WS_AUTH_TOKEN` accepted a
native WebSocket with no Origin. That client could select supervised mode,
create a bounded negative fader/gain proposal and reach the physical manual
write path. dLive and Mixing Station direct handlers exposed a shorter variant.

Electron also passed its bearer in the backend process environment. Removing
the variable from Python did not remove it from the process environment visible
to other processes owned by the same macOS user.

## Decision

- A running source backend without a configured token is
  `unauthenticated_read_only`.
- Its ingress is deny-by-default before dispatch. Only the immutable explicit
  read/query, meter/local-audio and emergency-stop allowlist is accepted;
  future `get_*` names do not gain authority by prefix.
- `connect_wing` is allowed without a token only when the request explicitly
  sets `observation_only=true`.
- Network startup forces Analyze mode, disables auto unlock and forces WING
  observation-only when no token is configured.
- The attended M005 runner creates an ephemeral token, passes it through a
  one-shot inherited pipe and authenticates before sending its first recorded
  proof request. The bearer is absent from both child environment and proof
  event stream. Client id, TTL and Origin policy are deterministic rather than
  inherited from the operator shell.
- The M005 child receives a minimal environment with an exact repository
  `PYTHONPATH`; the bound launch nonce disables repo `.env` loading entirely.
  Untracked operator-mode, write-boundary or evidence-root values therefore
  cannot influence a release-bound proof without appearing in provenance.
- Electron passes its token through inherited file descriptor 3 and closes the
  pipe after one write. The backend consumes and closes that descriptor before
  constructing the authentication manager.
- The Electron FD token lives for exactly the backend process lifetime rather
  than expiring after 24 hours. Finite CLI tokens are checked both before FIFO
  enqueue and again before dispatch, so a buffered control frame cannot run
  after expiry; the emergency fast lane remains available.
- Token pipes have one deadline covering the complete payload through EOF;
  partial writers cannot hang backend startup.
- Repo `.env` files cannot provide either launch-binding values or a plaintext
  WebSocket bearer; an explicit process environment remains available for the
  controlled CLI compatibility path.
- Development Electron owns and starts its own Vite server on an OS-assigned
  `127.0.0.1` port. It never trusts a fixed port that another local process can
  pre-bind before bootstrap-token IPC becomes available.
- Closing the last Electron window quits on every platform. Quit clears the
  token, sends SIGTERM to the backend, waits for child exit and escalates to
  SIGKILL rather than leaving authenticated controller authority headless.
  Child-process `error` events trigger escalation; they are not treated as
  proof that the backend exited. Quit completes only after `close` or an OS
  check confirms that the child no longer exists.

## Why this won

The boundary is enforceable before any handler-specific policy and new message
types default to blocked. Read-only diagnostics remain available without
turning a local unauthenticated socket into console authority.

## Rejected alternatives

- Trusting loopback or `Origin=None`: any local process can satisfy it.
- Keeping the bearer in an environment variable: same-user process inspection
  can recover it.
- Blocking emergency stop for unauthenticated clients: loss of a safety action
  is worse than allowing an idempotent freeze request.

## Test plan

- Forged mode/proposal/apply/direct-write and disconnect messages are blocked
  before dispatch without a token.
- Read queries, meters and explicit observation-only WING connect remain
  available.
- Authenticated sessions retain their existing supervised path.
- The M005 runner authenticates without recording the bearer.
- Electron environment contracts contain only the inherited FD number, and a
  process smoke confirms the token crosses FD 3 without entering child env.
- A hostile listener on port 3000 cannot become the Electron renderer.
- Unknown future `get_*` frames are blocked, every allowlisted name exists in
  dispatch, a partial token pipe times out, and `.env` cannot activate control.
- A poisoned repo `.env` is ignored by the M005-bound child, an expired buffered
  control frame is never dispatched, and last-window close retains no backend.

## Risks and rollback

Standalone browser development is read-only unless launched through secured
Electron or another token-authenticated launcher. Do not roll back to implicit
loopback trust; add an explicit secure launcher if another operator workflow is
needed. Same-user native denial of service remains outside the trust boundary:
a local process can consume the two one-second pending-auth slots, just as it
can terminate or suspend the operator process. This residual grants no token,
readback, or mixer authority and is recorded rather than misrepresented as
cross-user isolation.
