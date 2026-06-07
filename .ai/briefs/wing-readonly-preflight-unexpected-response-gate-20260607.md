# Brief: WING Read-Only Preflight Unexpected Response Gate

## Context

The first supervised WING MVP blocker is still real WING read-only connection
proof. The runner already writes a strict event log and the evidence reviewer
rejects non-strict logs, but the real run path should fail as early as possible.

## Problem

`send_and_wait` could receive an unrelated backend event, keep waiting for the
expected response, and still let the run continue. The later evidence reviewer
would reject the artifact because the event log was not strict send/recv, but
the runner itself did not stop at the first unexpected response.

## Scope

- Make each request accept exactly one response of the expected type.
- Block the run on an unexpected response before sending later requests.
- Record the unexpected response in `summary.json`.
- Keep the change read-only and transport-free in tests.

## Out Of Scope

- Running real WING hardware.
- Closing the read-only blocker from offline evidence.
- Changing supervised write behavior.
