# M006 Owner Readiness Decision Verifier

## Context

The M006 owner readiness packet prepares review inputs for the narrow CH3
fader and gain-down WING proof artifacts. It intentionally remains
report-only: it does not close M005/M006, does not update the readiness matrix,
and does not grant live authority.

That leaves one offline safety gap before a future matrix update: an owner
decision artifact could accidentally accept wider surfaces such as EQ, pan,
mute, routing, fader boosts, gain-up, or full-auto apply.

## Options considered

1. Let the owner update `backend/safety_coverage_matrix.py` manually after
   reading the packet.
2. Reuse `tools/verify_m006_owner_readiness_packet.py` and add owner-decision
   fields to the packet verifier.
3. Add a separate offline decision verifier that consumes the packet, the
   ready packet-verifier artifact, and the owner decision artifact.

## Decision

Add `tools/verify_m006_owner_readiness_decision.py` plus a pending owner
decision template next to the M006 packet artifact.

## Why this won

The decision verifier keeps responsibilities separate:

- the packet verifier proves review input completeness after M005;
- the owner decision records human acceptance;
- the decision verifier checks that acceptance is limited to
  `fader_supervised` and `gain_supervised`.

It also preserves the existing safety invariant that verifiers are offline and
cannot grant live authority by themselves.

## Rejected alternatives

- Manual-only matrix editing was rejected because it has no machine-checkable
  guard against accepting extra surfaces.
- Extending the packet verifier was rejected because packet readiness and owner
  acceptance are different gates with different failure modes.

## Implementation plan

1. Add `owner_decision_template.json` with `pending_owner_review`.
2. Add a decision verifier that requires:
   - decision schema `automixer.m006_owner_readiness_decision.v1`;
   - an accepted packet-verifier summary with `failed_checks=[]`;
   - M005 summary path binding;
   - exactly `fader_supervised` and `gain_supervised`;
   - exactly `fader` and `gain`;
   - denied live authority, transport, direct-write, fader-boost, gain-up, and
     full-auto flags.
3. Add focused tests for pending, rejected, accepted, widened-scope, bad
   packet-verifier, path mismatch, and authority-claim cases.

## Test plan

Run:

```bash
PYTHONPATH=backend python3 -m py_compile tools/verify_m006_owner_readiness_decision.py tests/test_verify_m006_owner_readiness_decision.py
PYTHONPATH=backend python3 -m pytest tests/test_verify_m006_owner_readiness_decision.py -q
PYTHONPATH=backend python3 -m pytest tests/test_verify_m006_owner_readiness_packet.py tests/test_verify_m006_owner_readiness_decision.py -q
```

Before merge, also run the WING MVP rails after releasing the writer lease.

## Risks and rollback

Risk: the template may be mistaken for an accepted decision.

Mitigation: the template uses `pending_owner_review`; the verifier returns
`blocked` until an accepted decision, ready packet-verifier summary, and M005
path binding are present.

Rollback: remove the verifier, tests, template path, and this ADR. The prior
packet verifier remains report-only and safe.
