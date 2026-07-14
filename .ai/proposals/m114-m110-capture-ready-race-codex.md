# Codex Proposal — M114 M110 Capture-Ready Race

## 1. Short thesis

Treat `start_audio_capture_preview` as ownership/identity establishment and
allow only callback freshness to settle through a bounded same-session status
poll before HW01 route proof.

## 2. What I understood

The real WING stream starts, but `AudioCapture.get_status()` may legitimately
return `real_input=false` until its first CoreAudio callback. The current runner
requires `real_input=true` in the immediate start response, so scheduler timing
can block an otherwise valid device. Simply accepting `running=true` would be
unsafe because route/headroom proof requires actual fresh samples.

## 3. Proposed solution

- Split preview validation into stable identity/ownership checks and the final
  freshness predicate.
- Require the first terminal to prove that this WebSocket owns a running
  `sounddevice` preview for the requested exact index, a non-empty device name,
  48 channels and 48 kHz.
- If `real_input` is not yet true, send only
  `get_audio_capture_status` on the same WebSocket at a fixed short interval,
  for at most one second.
- Pin device index, name, source type, channel count and sample rate across
  every poll. A stopped stream, error, identity drift, malformed response or
  deadline fails closed and invokes existing teardown.
- Continue to the route request only after fresh real input is observed. The
  backend route gate independently rechecks ownership, callback age, sample
  progression, serial/USB binding, Main 47/48 and headroom.

## 4. Likely files to touch

`tools/wing_attended_soundcheck_runner.py`,
`tests/test_wing_attended_soundcheck_runner.py`, M114 Council records, ADR and
plan, current truth surfaces after validation, and finally only the two CP05
marker files in a separate rebind commit.

## 5. Alternatives considered

Retrying the whole runner leaves a nondeterministic race. Sleeping once is
fragile and does not observe identity drift. Waiting inside the shared audio
handler changes every preview consumer. Accepting `running=true` alone weakens
the real-input contract. A bounded runner-owned poll is the narrowest fix.

## 6. Risks

An unbounded or overly permissive poll could hide a dead callback or device
swap. Extra status messages could also interleave with control terminals. The
existing terminal receiver and exact identity pinning keep both fail closed.

## 7. Test plan

Add tests for delayed first callback success, never-ready timeout, stopped
stream, source/device/channel/sample-rate drift and no route request before
readiness. Run the focused attended-runner and server tests, WING/static rails,
standard full suite, independent review, then a marker-only CP05 rebind and a
new route-only hardware attempt.

## 8. Where the other agent may disagree

The likely disagreement is whether one second is enough and whether readiness
should be polled in the handler. One second is already the backend's maximum
callback-age window and is long relative to the CoreAudio block period; keeping
the wait in the attended runner avoids changing general preview semantics.
