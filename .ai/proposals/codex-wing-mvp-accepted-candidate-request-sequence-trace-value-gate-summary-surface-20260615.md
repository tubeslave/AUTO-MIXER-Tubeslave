# Proposal: Accepted-Candidate Request-Sequence Trace Value-Gate Summary Surface

## Short Thesis

Keep the existing no-WING proof-plan gate as the authority for accepted-candidate
request-sequence trace readiness, and document the surface with validation
evidence instead of widening live behavior.

## What I Understood

The WING MVP is still blocked on a real read-only preflight artifact. Offline
packets can prepare the operator path, but they must not imply readiness. The
request-sequence trace is one of the required accepted-candidate contracts that
must be proven before supervised fader/gain proof can even be planned.

## Proposed Solution

Use the existing `tools/no_wing_release_check.py` value-gate summary-surface
check as the implementation surface. The check records the required request
sequence gate fields, recomputes whether the summary surface matches the next
proof plan, and fails closed if a mismatch appears.

## Likely Files To Touch

- `.ai/briefs/wing-mvp-accepted-candidate-request-sequence-trace-value-gate-summary-surface-20260615.md`
- `.ai/proposals/codex-wing-mvp-accepted-candidate-request-sequence-trace-value-gate-summary-surface-20260615.md`
- `Docs/adr/wing-mvp-accepted-candidate-request-sequence-trace-value-gate-summary-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_ACCEPTED_CANDIDATE_REQUEST_SEQUENCE_TRACE_VALUE_GATE_SUMMARY_SURFACE_20260615.md`

## Alternatives Considered

- Add a new checker: rejected because the existing no-WING proof-plan check
  already owns the next proof-plan contract.
- Treat request-sequence absence as a manual note only: rejected because it
  would weaken the supervised prestart gate.
- Attempt a live WING run now: rejected because the current evidence says a real
  operator-captured read-only preflight candidate is still required.

## Risks

- The surface can become noisy because it repeats many proof-plan fields.
- Operators may mistake a passing no-WING value gate for real hardware proof.
  The summary must keep readiness closure and live authority false.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py tests/test_no_wing_release_check.py`
- Focused request-sequence and expected-artifact value-gate summary-surface tests.
- Temporary no-WING quick artifact plus manifest verification.
- Focused WING/no-WING suite and full pytest before commit.

## Where The Other Agent May Disagree

Kimi may prefer stopping at the real WING capture step instead of committing
another offline surface. I agree that hardware proof is the next durable
milestone; this package is acceptable only because it documents an already
implemented fail-closed gate and preserves the live boundary.
