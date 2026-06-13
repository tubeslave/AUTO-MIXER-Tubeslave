# Proposal: WING MVP Backend Listener Confirmation Template

## Short thesis

Generate a pending backend-listener confirmation template during busy-port
fail-closed no-WING runs, but require explicit operator edits before it can be
accepted.

## What I understood

The first MVP blocker is still WING read-only connection proof. A local listener
on `127.0.0.1:8765` can be legitimate if it is the intended backend, but the
gate must not assume that. It should help the operator preserve exact PID and
command-line evidence without granting authority.

## Proposed solution

- Add a helper in `tools/no_wing_release_check.py` that writes
  `backend-listener-confirmation-template.json` when the backend-port gate
  fails closed on an existing listener.
- Seed the template with current listener PID/command-line diagnostics.
- Set `status: operator_review_required` and
  `confirms_intended_backend: false` so the template remains invalid until
  edited.
- Surface the template path/status/safety flags in summary details and Markdown.
- Update runbook, MVP rails, Project Director evidence, and tests.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/`

## Alternatives considered

- Keep the manual JSON only: safe, but error-prone when copying PID and command
  line out of fail-closed summaries.
- Auto-generate an accepted confirmation: rejected because it would turn a
  diagnostic into an authority grant without operator review.

## Risks

The template could be mistaken for an accepted artifact. The mitigation is to
use a non-accepted status, false confirmation, explicit required edits, and
tests that prove it remains invalid until edited.

## Test plan

- Unit tests for template payload safety fields.
- Unit tests proving the pending template fails confirmation validation.
- no-WING release-check tests proving busy-port summaries include the template.
- MVP rail and Project Director tests for evidence visibility.

## Where the other agent may disagree

Kimi may argue the extra artifact adds another document to review. The
counterpoint is that it reduces manual transcription error exactly at the
current WING blocker without broadening live scope.
