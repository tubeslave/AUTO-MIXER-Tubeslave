# Proposal: Offline Fail-Closed Remediation

## 1. Short thesis

Replace false success and synthetic runtime evidence with measured, typed,
fail-closed behavior while keeping the current live-authority boundary at zero.

## 2. What I understood

The user requested every repair that is currently possible without a real
console. The worktree already contains unrelated M123 and dLive work, so this
change must remain isolated and must not rewrite those changes.

## 3. Proposed solution

1. Verify gain using a real getter and reject every unsupported action readback.
2. Return no audio devices when PyAudio is unavailable or fails; always
   terminate an initialized PyAudio instance.
3. Keep registered status routes for compatibility but return a typed
   unavailable, read-only, non-authoritative payload.
4. Reject duplicate WebSocket message types during registry construction.
5. Make agent preflight/evaluation inspect actual inputs and artifacts.
6. Disable fake MuQ genre scoring and require measured spectral evidence for
   legacy EQ proposals.

## 4. Likely files to touch

`backend/auto_soundcheck_engine.py`, `backend/audio_devices.py`, handler/server
modules, legacy EQ, agent-ops modules, MuQ experiment, and focused tests.

## 5. Alternatives considered

- Removing registered routes would be more disruptive to existing clients.
- Implementing every missing DSP feature would invent unverified behavior and
  exceed the no-console evidence boundary.
- Keeping explicit mocks behind an environment flag would retain a production
  confusion surface; test fixtures can inject mocks directly instead.

## 6. Risks

- Clients may have relied on fake device rows.
- Duplicate detection may expose a future collision at startup.
- `muq_genre` callers now receive an explicit error until a real scorer exists.

## 7. Test plan

Run focused regression tests, relevant server/soundcheck/handler suites, the
full backend suite, frontend tests/typecheck/build, `git diff --check`, and the
no-WING gate with the writer lease released.

## 8. Where another reviewer may disagree

A reviewer may prefer removing unavailable routes or adding explicit mock-mode
configuration. This proposal favors backward-compatible response types and no
synthetic production discovery.
