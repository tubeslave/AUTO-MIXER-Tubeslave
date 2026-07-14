# M114 M110 Capture-Ready Race

## Objective

Close the fail-closed startup race observed during the first real M110-HW01
attempt: the WING CoreAudio stream starts successfully, but the runner checks
`real_input` before the first callback and tears down the session before route
evidence can be collected.

## Real evidence

- macOS exposes one BEHRINGER WING at PortAudio index 1, 48 inputs, 48 outputs
  and 48 kHz; ioreg reports VID 5015, PID 1291, USB serial
  `0100Q1V0604AAE214` and location `0x00100000`.
- The CP05-bound route-only attempt on final HEAD `38f677752` connected to
  WING `192.168.1.102`, read channel names and started the real 48-channel
  `sounddevice` stream.
- It stopped with
  `m110_real_48_channel_capture_start_failed:terminal_contract_mismatch`.
  The backend log records `AudioCapture started (sounddevice)` immediately
  before clean disconnect and shutdown.
- The runner summary at
  `~/.automixer/runtime/m110_attended/plan-summary.json` records
  `live_write=false`, `accepted=false` and `readiness_closure_allowed=false`.
  No route mutation, reset or correction occurred.

## Scope

1. Preserve strict validation of the initial preview ownership and the pinned
   WING 48-channel identity.
2. When only callback freshness is pending, poll
   `get_audio_capture_status` on the same authenticated WebSocket for a short,
   fixed deadline.
3. Accept readiness only when the same device index/name, source, channel
   count and sample rate report fresh real input; every drift, stop, error,
   timeout or unexpected terminal fails closed and triggers teardown.
4. Add success, timeout and identity-drift tests.
5. Revalidate, commit the non-marker change, perform a separate marker-only
   CP05 rebind, then repeat HW01 before any live correction.

## Hard boundaries

- No bypass or extension of M110 authority.
- No retry loop without a finite deadline.
- No console writes during implementation or validation.
- No full attended execution until CP05 is rebound to the committed fix and
  HW01 route/headroom evidence succeeds.
- Preserve the main checkout's generated evidence.
