# Codex proposal: standard scan remediation

## 1. Short thesis

Close the 13 findings with fail-closed boundary fixes: deny authority where the
underlying protocol cannot authenticate, require the launch token and resource
ownership, enforce limits before allocation, and make evidence describe only
what actually happened.

## 2. What I understood

The highest-risk defects are trust-boundary errors rather than missing parsing
polish. UDP source tuples and loopback ports do not authenticate a mixer.
Tokenless local clients are still untrusted. Offline simulation and rendering
must not mint `applied` evidence after a failure or for an in-memory transition.

## 3. Proposed solution

1. Mark WING OSC receipts observational and reject them as authoritative proof.
2. Reject WebSocket sessions when no valid launch token is configured; bind a
   generic mixer transport owner to the authenticated requesting socket.
3. Quarantine Mixing Station as observational until authenticated IPC exists.
4. Enforce file, line, sample, and aggregate limits while reading JSONL.
5. Propagate requested DSP failures and make replay events explicitly simulated.
6. Validate all Safety Gate numeric configuration as finite and range-safe.
7. Bind Virtual Mixer services to loopback and reject malformed/non-finite levels.
8. Bound GPT bridge sockets, response/file reads, and pre-auth concurrency.

## 4. Likely files to touch

`backend/wing_client.py`, `backend/server.py`,
`backend/handlers/connection_handlers.py`, ML loaders/service,
`ai_mixing_pipeline` renderer/replay, `automixer/safety/safety_gate.py`,
`virtual_mixer`, `tools/gpt_control_bridge/server.py`, and focused tests.

## 5. Alternatives considered

Cryptographic WING/Mixing Station challenges were rejected because the reviewed
protocols do not provide a trustworthy response binding. A new capability
broker was rejected as too broad for this remediation.

## 6. Risks

Source-mode clients without the packaged launch token will no longer connect.
WING supervised mutations and Mixing Station server integration will fail
closed until an authenticated proof channel exists. This is intentional and
matches the current no-authority milestone.

## 7. Test plan

Targeted exploit regressions plus positive controls, owning suites, no-WING
quick, full pytest, import/compile checks, and diff hygiene. No hardware tests.

## 8. Where the other agent may disagree

An alternate review may prefer unauthenticated emergency-stop availability or
observational Mixing Station state. Those choices would retain the reported
availability and provenance vulnerabilities unless separated behind a new,
stronger local capability boundary.
