# Independent patch review 2 — M120

## Verdict

ACCEPT.

## Verification

- Disconnect invalidation is accepted only after a valid
  `connection_status/connected=false` acknowledgement.
- The accepted notification requires the backend's disconnect-specific
  message, both running flags false, a readiness mapping and no failed/error
  result.
- Emergency-stop, watchdog, physical-console-change, pre-ack, failed and
  errored invalidations fail closed.
- The normal active-session receiver remains unchanged and strict.
- Disconnect-local invalidation evidence is separate from a prior readiness
  latch.
- Duplicate acknowledgement/invalidation and unrelated control terminals
  remain fatal.
- The ADR matches the revised implementation and tests.

The reviewer independently repeated the full runner test file with
`142 passed` and confirmed `git diff --check` is clean. No blocking or
significant remaining findings were reported.
