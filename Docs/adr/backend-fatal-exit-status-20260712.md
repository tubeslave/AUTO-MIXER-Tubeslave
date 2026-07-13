# Backend fatal exit status

## Context

`backend/server.py` logged exceptions raised by `AutoMixerServer.start()` but
then returned from the process with status `0`. A port collision reproduced the
failure: the backend was dead, yet a supervisor, Electron launcher or CI runner
could classify the shutdown as successful. The same path also called cleanup
once in `finally` and again through the registered `atexit` callback.

## Options considered

1. Keep log-only failure reporting.
2. Re-raise every startup exception before cleanup.
3. Make the entrypoint return an explicit status, always clean up, clear the
   global atexit owner and exit through `SystemExit`.

## Decision

Use option 3. `main()` returns `1` for fatal startup/runtime or cleanup errors,
`0` for a normal shutdown or operator `KeyboardInterrupt`, and the module
entrypoint raises `SystemExit(main())`.

## Why this won

It preserves deterministic cleanup while giving process supervisors a truthful
failure signal. Clearing the global server reference prevents duplicate atexit
cleanup after the normal `finally` path.

## Rejected alternatives

Log-only handling was rejected because logs are not a machine-readable liveness
contract. Immediate re-raise was rejected because it could skip controlled
cleanup or obscure a cleanup failure.

## Implementation plan

1. Move environment parsing, construction and `asyncio.run()` into `main()`.
2. Convert caught fatal exceptions to exit status `1`.
3. Treat cleanup failure as fatal and clear `_server_instance` after cleanup.
4. Add unit and real occupied-port process regressions.

## Test plan

- Targeted entrypoint and port-collision tests in `tests/test_server.py`.
- Real subprocess with a pre-bound loopback port must return `1` and clean up
  once.
- Full backend suite and no-WING gate before handoff.

## Risks and rollback

Supervisors that incorrectly relied on status `0` after a fatal crash will now
restart or fail the job, which is the intended behavior. Rollback is a source
revert; do not restore log-only fatal handling.
