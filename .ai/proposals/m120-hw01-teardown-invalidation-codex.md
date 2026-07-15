# Codex proposal — M120 HW01 teardown readiness invalidation

## 1. Short thesis

Add a teardown-only disconnect receiver in the proof-owned runner. It should
accept the backend's bounded, semantically expected readiness-invalidation
notification while retaining the strict normal terminal receiver for every
active-session operation.

## 2. What I understood

The real WING route proof succeeded and produced a fully accepted artifact.
The backend then disconnected correctly and broadcast `connection_status`
with `connected=false`, followed by `auto_soundcheck_status` with
`status=readiness_invalidated`. The generic terminal receiver performs a short
duplicate-terminal grace read after the expected response and correctly treats
readiness invalidation as fatal during active work. Reusing that receiver for
terminal teardown therefore converts an expected disconnect side effect into
`wing_disconnect_failed:RuntimeError`.

## 3. Proposed solution

Implement a private teardown-only receiver that:

- waits for exactly one `connection_status` terminal;
- requires `connected is False`;
- permits at most one exact
  `auto_soundcheck_status/readiness_invalidated/running=false` notification
  before or during the duplicate grace window;
- rejects duplicate connection terminals, any other control terminal, blocked
  or failed payload, and malformed invalidation payloads;
- records whether the expected invalidation was observed for teardown evidence.

Use it only in `ProofOwnedBackendSession._close_impl()`. Do not relax
`_receive_terminal()` or `_reject_unexpected_terminal()`.

## 4. Likely files to touch

- `tools/wing_attended_soundcheck_runner.py`
- `tests/test_wing_attended_soundcheck_runner.py`
- M120 Council/ADR/plan and required current-truth surfaces

No production dependency is required.

## 5. Alternatives considered

- Suppress the backend readiness-invalidated broadcast: rejected because the
  broadcast is truthful and may be required by other clients.
- Catch every teardown `RuntimeError`: rejected because it could hide a real
  disconnect, ownership or protocol failure.
- Accept the route artifact despite unverified teardown: rejected because M110
  explicitly requires clean teardown before HW01 closure.
- Remove duplicate-terminal checking globally: rejected because it weakens the
  one-terminal protocol rail during active work.

## 6. Risks

- Overly broad matching could hide a real safety invalidation. Mitigate with
  exact type/status/running checks and teardown-only call placement.
- An invalidation arriving before the disconnect terminal could be lost or
  misclassified. The teardown receiver should allow one exact notification on
  either side while still requiring the disconnect terminal by deadline.
- Multiple invalidations could indicate duplicate dispatch. Reject the second.

## 7. Test plan

- Positive: disconnect terminal followed by exact readiness invalidation.
- Positive: exact readiness invalidation followed by disconnect terminal.
- Negative: duplicate connection terminal.
- Negative: duplicate readiness invalidation.
- Negative: readiness invalidation with `running=true` or wrong status.
- Negative: unrelated blocked/error/control terminal.
- Existing M110 runner suite, route-proof suite and WING rails.
- Full pytest and no-WING release check before CP05 rebind.
- Repeat real-WING route-only HW01 after the marker-only commit.

## 8. Where the other agent may disagree

The independent reviewer may prefer changing server broadcast ordering or
adding a request-correlated disconnect terminal. Those would be broader
protocol changes. The teardown-only receiver is smaller and preserves current
server semantics, but the reviewer should pressure-test its exact acceptance
envelope and duplicate handling.
