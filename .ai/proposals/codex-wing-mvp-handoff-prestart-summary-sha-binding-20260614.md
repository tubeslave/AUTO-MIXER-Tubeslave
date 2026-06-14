# Proposal: Bind handoff prestart summary by SHA256

## Short thesis

Add a digest binding to the WING read-only handoff packet so a real prestart
summary path cannot be silently swapped between no-WING plan creation and the
operator handoff packet.

## What I understood

The first open MVP blocker is still WING read-only connection proof. The current
rail requires a no-WING prestart summary before the real backend/WING session,
and accepted review candidates later require that summary by path and SHA256.
The handoff packet currently exposes the path but not the digest.

## Proposed solution

- Hash an existing prestart summary file in
  `tools/build_wing_readonly_preflight_handoff_packet.py`.
- Store `backend_port_prestart_summary_sha256` plus an availability flag in the
  packet.
- In `tools/no_wing_release_check.py`, compare the packet digest to the plan
  artifact digest and fail closed on mismatch.
- Render the digest and match flag in no-WING summary output.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- WING MVP docs, report, and ADR files.

## Alternatives considered

- Leave SHA only in the plan artifact. Rejected because the operator handoff is
  the visible bridge into the hardware step.
- Require a SHA even for placeholder packets. Rejected because standalone packet
  generation before no-WING prestart should remain possible and clearly show no
  digest is available yet.

## Risks

The main risk is over-constraining packet generation. The implementation keeps
placeholder packets valid while making no-WING generated packets carry the
actual digest.

## Test plan

- Focused handoff packet tests.
- Focused no-WING validation tests.
- no-WING quick gate.
- Project Director and WING rail tests.

## Where the other agent may disagree

Kimi may argue the plan artifact already contains the digest. The counterpoint
is that the operator-visible handoff packet is the object used at the real
session boundary, so it should carry the same binding.
