# Proposal: WING MVP Handoff Invalid-Record Marker

## Short Thesis

The handoff packet should expose a direct boolean proving that supervised
proof-plan required fields include invalid review-decision records.

## What I Understood

The WING MVP is still blocked on real read-only WING evidence and manual
operator review. The invalid-record proof-plan field is now part of the
contract, but the handoff packet should make its presence easy to verify before
operators consider supervised proof.

## Proposed Solution

- Add `required_supervised_proof_plan_summary_fields_include_invalid_records`
  to the handoff packet.
- Render the flag in handoff markdown.
- Validate the declared flag in no-WING handoff artifact checks.
- Propagate the flag into no-WING summary JSON and markdown.
- Add Project Director and WING rails coverage so Current First Command cannot
  omit it.

## Likely Files To Touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/MVP_WING_SUPERVISED.md`

## Alternatives Considered

- Rely on the full required-field list only. Rejected because the operator must
  manually scan a long list.
- Add no new field and only extend docs. Rejected because no-WING should
  fail closed when the handoff packet's declared marker is stale.

## Risks

Risk is limited to offline report shape. A stale handoff packet now fails
validation instead of silently hiding the missing invalid-record marker.

## Test Plan

Run targeted handoff/no-WING/Project Director/WING-rails tests, no-WING quick,
Project Director JSON, diff check, memory lint, and standard pytest.

## Where The Other Agent May Disagree

Kimi may prefer deriving all boolean handoff markers from the field contract.
That is reasonable later; this patch matches the existing explicit-marker
handoff style.
