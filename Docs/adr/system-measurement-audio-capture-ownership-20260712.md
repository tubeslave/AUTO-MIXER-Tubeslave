# System Measurement Audio Capture Ownership — 2026-07-12

## Context

Meter preview and system measurement shared `AutoMixerServer.audio_capture`
without serialization. A second client could reconfigure or stop the device
during a measurement. The analyzer accepted as little as half a second even
when the operator requested six seconds, so stale or truncated buffers could
produce correction evidence. Synchronous sounddevice stop/close could also
block the asyncio loop for up to two seconds.

## Decision

- All preview start/stop and system-measurement device operations share one
  asyncio lock.
- AutoSoundcheck reserves the same ownership boundary and always owns its
  private capture. It is rejected while preview/measurement capture exists;
  measurement is rejected while AutoSoundcheck is active.
- Device start/reconfigure/stop and transfer-function analysis run off the event
  loop through the cancellation-draining worker helper.
- A measurement may reuse only the initiating WebSocket's preview. A preview
  owned by another client fails with an explicit in-use result.
- Only one measurement task may run at a time.
- Measurement requires distinct positive channels, a bounded duration, a fresh
  real-input callback, the full number of newly captured callback samples and
  complete buffers for both channels.
- One data lock spans every channel write in a callback generation. Reference
  and microphone buffers are copied through one atomic multi-channel snapshot,
  preventing a one-block cross-channel skew or torn ring read.
- A measurement-created capture is stopped in `finally`; a requester-owned
  preview remains running.
- Graceful shutdown cancels and drains an active measurement before controller
  cleanup, then drains AutoSoundcheck and AudioCapture off the event loop.
  Emergency stop latches immediately and drains a slow AutoSoundcheck stop in a
  worker while the event loop remains responsive. A cancelled capture start
  stops any newly created but not-yet-owned device before releasing the lock.
- AutoSoundcheck always finalizes owned mixer/audio resources, including
  connection failure, audio-start failure and later pipeline exceptions. A
  live worker or monitor thread keeps the engine active even if its state is
  `error`; `stopped` is published only after join and cleanup complete.
- AudioCapture stop failures and a thread that survives the join timeout are
  surfaced as incomplete teardown and retain retryable references. Graceful
  shutdown accumulates failures but always closes clients, runs remaining
  cleanup and releases the shutdown event; the process then exits nonzero.
- Preview cancellation distinguishes worker success from worker failure: after
  a drained successful stop it releases the exact owner/capture even when the
  caller was cancelled, while a failed stop retains both references for retry.
  A failed capture start likewise retains a live stream/thread rather than
  erasing evidence of an orphaned device.

## Why this won

The change preserves continuous meters for the normal single-operator flow but
prevents cross-client capture replacement. A result now proves the requested
fresh duration instead of merely proving that a ring buffer contained some
audio at some point.

## Rejected alternatives

- Always reuse the global capture: ownership and freshness remain ambiguous.
- Always reject measurement while preview runs: the normal cockpit starts a
  preview automatically and would make measurement unusable.
- Stop sounddevice directly in async handlers: it delays WebSocket processing,
  including the emergency fast lane.

## Test plan

- Requester preview reuse and measurement-owned cleanup.
- Cross-client and concurrent measurement rejection.
- Bidirectional AutoSoundcheck/measurement and preview/AutoSoundcheck rejection.
- Atomic reference/microphone snapshot during an intentionally interleaved
  callback.
- Stale input and truncated-buffer rejection.
- Non-blocking preview, emergency and shutdown stop plus cancellation-draining
  cleanup.
- Repeated controller cleanup stops every resource once.
- Early AutoSoundcheck failure cleanup, stop-timeout restart rejection,
  cancelled-start orphan cleanup and fail-closed shutdown truth.

## Risks and rollback

Future audio-consuming runtimes must reserve the same ownership boundary before
opening a device. The lock protects lifecycle ownership; the separate data lock
protects per-callback channel alignment. Neither is optional for new capture
consumers.
