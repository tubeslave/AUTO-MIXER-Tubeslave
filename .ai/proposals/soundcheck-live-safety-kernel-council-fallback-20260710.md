# Council fallback proposal: soundcheck live safety kernel

Kimi CLI was requested in read-only plan mode but returned HTTP 402 during
membership verification. A separate read-only council agent was also stopped
after failing to return within the review window. This proposal records the
already completed independent live-safety and product audits used as fallback;
it does not claim a new Kimi result.

## 1. Short thesis

Create a real single `ConsoleWriteBroker`; do not rely on convention around the
existing manual gate. Runtime allowlists must be compiled from readiness truth,
and every other component must be structurally unable to own a write transport.

## 2. What I understood

The critical failure is not DSP quality. It is the gap between declarative
proposal-only policy and executable broad writers. A working soundcheck product
can be shipped as read-only analysis plus supervised fader/gain-down while broad
families stay blocked.

## 3. Proposed solution

- Default to loopback and authenticate any remote WebSocket before dispatch.
- Remove client-controlled approval and auto-apply authority.
- Make AutoSoundcheck and AI/LLM proposal-only.
- Route the approved narrow write path through one broker with fresh pre-read,
  physical post-read, complete rollback and one emergency generation latch.
- Separate commanded values from observed console values.
- Generate runtime write allowlists from the canonical safety/readiness matrix.
- Treat disabled safety components as blocked, never bypassed.

## 4. Likely files to touch

The same runtime and test files as the Codex proposal, plus a dedicated broker
module after the existing path has been made fail-closed.

## 5. Alternatives considered

- Patch every writer separately: rejected because future bypasses remain easy.
- Enable broad AutoSoundcheck with pre-read/readback: rejected because most
  families have no accepted hardware proof and multi-field rollback is partial.

## 6. Risks

- Calling the existing manual gate the single writer without removing raw
  transports leaves the architecture vulnerable.
- Cache fallback and default getter values can fabricate rollback snapshots.
- Remote auth that is optional on non-loopback binds is not sufficient.

## 7. Test plan

Add negative tests for every bypass, static raw-write ownership checks, network
loss/retry races, field-by-field readback, emergency cancellation and rollback
compare-and-swap before hardware proof.

## 8. Where Codex may disagree

Codex favors hardening the existing manual gate first. Council accepts that as
the smallest first checkpoint only if raw writers are simultaneously forced into
proposal-only mode and a later extraction into a broker remains explicit.
