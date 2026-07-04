# WING M005 Disconnect Timeout Proof Runner

## Context

M005 needs reproducible evidence that the backend fails closed when WING
connectivity is lost, stale physical readback is unavailable, and a supervised
blocked-write probe is attempted. A prior scratch proof used the wrong
WebSocket request type, so it did not disconnect the mixer before a real
supervised fader move was approved and then rolled back.

The proof must stay narrower than live authority. It may exercise only the
existing Product Layer proposal path, only after accepted read-only preflight
review, and only after strict evidence that fresh physical readback is
unavailable.

## Options considered

1. Keep using ad hoc WebSocket scratch scripts for M005.
2. Extend the supervised apply runner with disconnect-specific branches.
3. Add a dedicated M005 disconnect/timeout/stale-state proof runner.

## Decision

Use option 3. Add `tools/wing_timeout_disconnect_proof_runner.py` as a
dedicated M005 runner. The runner allows only canonical backend disconnect
request types, requires physical baseline readback, requires accepted read-only
preflight decision review before `--execute`, and sends a blocked-write probe
only after disconnect and unavailable-readback checks pass.

## Why this won

A dedicated runner keeps the M005 safety order explicit:

1. verify physical baseline readback,
2. create a proposal,
3. confirm software disconnect or operator-confirmed physical cable loss,
4. verify stale/live physical readback is unavailable,
5. send `apply_operator_proposal` only as a blocked-write probe,
6. rollback and require manual review if the probe applies unexpectedly.

This prevents the earlier wrong-message defect from becoming reusable proof and
keeps disconnect evidence separate from normal supervised apply proof.

## Rejected alternatives

- Ad hoc scratch scripts were rejected because they can use invalid request
  types or skip the pre-write unavailable-readback gate.
- Extending the supervised apply runner was rejected because the normal proof
  runner proves a successful supervised write, while M005 proves loss-state
  fail-closed behavior.

## Implementation plan

- Add a dedicated M005 CLI runner with `software` and
  `physical-cable-loss` modes.
- Restrict software disconnect requests to `disconnect_mixer` and
  `disconnect_wing`.
- Require `--accepted-readonly-source-id` and the existing read-only decision
  gate before backend contact in `--execute`.
- Record summary fields that keep `live_authority_granted=false`,
  `readiness_closure_allowed=false`, and
  `readiness_matrix_update_allowed=false`.
- Record `required_m005_evidence`, `completed_m005_evidence`,
  `missing_m005_evidence`, `remaining_m005_evidence`, and `m005_complete`
  so a partial runner pass cannot be mistaken for complete M005 closure.
- Keep software-disconnect proof marked as leaving physical cable-loss and
  reconnect-recovery evidence still required.

## Test plan

- Compile the new runner and tests with `py_compile`.
- Fake-WebSocket test for canonical software disconnect followed by expected
  `mixer_not_connected` block.
- Fake-WebSocket test that live or stale physical readback after disconnect
  stops before any apply request.
- Validation test for physical cable-loss operator confirmation.
- Preflight gate test that blocks before opening the WebSocket.
- Unexpected apply test that sends rollback and marks manual review required.
- Summary-contract tests that dry-run and software-disconnect evidence keep
  `m005_complete=false`, while physical cable loss plus reconnect recovery is
  the only complete M005 artifact shape.
- Full repository pytest before landing.

## Risks and rollback

Risk: the runner still sends `apply_operator_proposal` in the final probe step,
so an unexpected backend regression could apply a move. The runner mitigates
this by allowing only negative fader/gain deltas, requiring unavailable
readback before the probe, and rolling back when an unexpected apply is
reported.

Rollback is a normal git revert of the runner, tests, and ADR. M005 should then
remain blocked rather than returning to ad hoc proof scripts.

## 2026-07-04 Contract Hardening Update

M005 completion now requires reconnect recovery to include a fresh physical
readback in the reconnect phase, not only `status=passed` and
`unchanged_from_baseline=true`. The M006 owner-readiness verifier also
independently checks the M005 phase details: physical cable-loss confirmation
token, unavailable stale-readback payload, safe blocked-write reason, connected
reconnect status and physical reconnect readback.

This keeps summary booleans from becoming sufficient owner-review evidence by
themselves. The update is offline verifier hardening only; it grants no WING
contact, no live apply authority and no readiness closure without the
operator-present M005 proof and owner M006 decision.

## 2026-07-04 Request Audit Binding Update

The M006 owner-readiness verifier now treats the M005 summary request audit as
part of the owner-review input contract. A complete M005 summary must expose
`sent_request_types`, must show the blocked `apply_operator_proposal` probe,
and must not contain direct forbidden request types such as `set_fader`,
`set_gain`, `route_output`, `load_snap` or `save_snap`. The verifier also
cross-checks `forbidden_request_types_sent` against the request audit list, so
manually clean summary flags cannot hide a direct write request.

This is still offline verifier hardening only. It validates the artifact
contract before owner review and grants no WING contact, live apply authority
or readiness closure.

## 2026-07-04 Event Log Audit Binding Update

The M006 owner-readiness verifier now cross-checks the M005 summary request
audit against the runner `events.jsonl` send rows from the same artifact run
directory. A complete M005 summary is not enough if the event log is missing,
malformed, points outside the summary run directory, differs from
`sent_request_types`, lacks the blocked `apply_operator_proposal` probe, or
contains a forbidden direct write request.

This keeps the append-only event stream as the authoritative request audit for
owner-review input. The update is offline verifier hardening only; it does not
run M005, does not contact the backend or WING, and does not grant readiness
closure or live authority.

## 2026-07-04 M005 Artifact Path Binding Update

The M006 owner-readiness verifier now also binds the supplied M005 summary to
the canonical M005 proof root. Owner-review input must use a repo-relative
`artifacts/wing_timeout_disconnect_proof/*/summary.json` path, the summary's
declared `summary_path` must match that input path, the declared `events_path`
must match the loaded event log, and the event log must be the sibling
`events.jsonl` in the same run directory.

This prevents temporary-root, copied or path-mutated M005 summaries from
becoming owner-review input. The update is offline verifier hardening only; it
does not run M005, does not contact the backend or WING, and does not grant
readiness closure or live authority.
