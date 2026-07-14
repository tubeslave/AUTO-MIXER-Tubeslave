# Independent Proposal — M114 M110 Capture-Ready Race

## 1. Short thesis

Keep ownership and exact stream identity mandatory in the first terminal, then
poll only callback readiness on the retained WebSocket for at most one second.

## 2. What I understood

The handler returns `capture.get_status()` immediately after starting the
stream, while `real_input` becomes true only after `_audio_callback` records a
callback timestamp. The real HW01 attempt therefore hit a valid transition,
not a failed stream. The current single-terminal runner contract cannot
distinguish those states.

## 3. Proposed solution

- The initial response must prove `running`, `preview` and
  `owned_by_requester`; set `capture_started` immediately after this ownership
  proof so every later failure explicitly stops the preview.
- Pin exact source type, device name/index, channel count and sample rate.
- Require a strict sample counter and, before route proof, `real_input=true`, a
  finite callback age in 0..1 seconds and positive sample progression.
- If readiness is pending, request only `get_audio_capture_status` on the same
  WebSocket until an absolute monotonic deadline no greater than one second.
- Any stop, error, malformed status, optional owner/preview false, identity
  drift, stale callback or timeout fails immediately and uses existing teardown.
- Reserve the terminal duplicate-detection grace inside the absolute deadline.

## 4. Likely files to touch

The attended runner, its focused tests, M114 Council/ADR/plan records, truth
surfaces after validation, and the CP05 marker pair only after the non-marker
commit.

## 5. Alternatives considered

Whole-session retries remain nondeterministic; a blind sleep proves nothing;
handler-wide waiting changes unrelated preview consumers; accepting only
`running` weakens the physical-input proof.

## 6. Risks

Polling without a hard deadline can hang authority construction. Polling too
quickly can hit WebSocket rate limits. A generic status does not repeat the
owner field, so ownership must be established by the initial terminal and
rechecked by the backend route gate. Setting `capture_started` too late can
skip explicit preview teardown after an identity failure.

## 7. Test plan

Cover immediate readiness with zero polls; delayed readiness; multiple pending
statuses; timeout; initial ownership and identity failures; drift of every
pinned field; stopped/error/malformed/bool-as-int statuses; stale callback; no
sample growth; cancellation teardown; hard deadline; and exact sent-message
ordering with no second start or route request before readiness.

## 8. Where the other agent may disagree

The main choice is poll cadence. The duplicate-terminal grace is not by itself
a safe cadence because the authenticated WebSocket has a lower sustained rate
limit. An explicit interval should keep the poll rate below that limit while
remaining inside the one-second budget.
