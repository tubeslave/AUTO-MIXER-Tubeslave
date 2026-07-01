# Correction Method Goal Audit

## Context

The active correction-method goal asks Automixer to provide a complete
proposal-layer for gain/trim, fader/balance, EQ, compressor, gate/de-esser,
FX/sends, pan/stereo, routing/groups/scenes read-only, and master/output
proposal-only work without a real mixer. Each method must expose evidence,
problem statement, proposed correction, confidence, bounded delta, safety
blockers, applicable headroom/feedback/true-peak checks, operator explanation,
rollback expectation, future WING readback contract and tests.

The existing operator-review packet contains the detailed method cards, but a
reviewer still needed a compact machine-checkable surface that answers whether
the active offline goal is covered without granting live authority.

## Options Considered

1. Treat the operator-review packet and export verifier as the only evidence.
2. Add a dedicated offline goal-audit tool and artifact.
3. Wait for real WING proof before recording correction-method coverage.

## Decision

Use option 2. The correction-method proposal-layer now has a dedicated
offline audit surface that checks objective-family coverage, expected method
cards, priority sections, operator surfaces, and authority-denial flags.

The audit can pass only as offline proposal-layer evidence. It cannot grant
live authority, queue import, readiness closure, WING contact, backend contact,
or mixer mutation.

## Why This Won

The dedicated audit gives a small, repeatable closure check for the user-visible
offline goal while keeping the project-completion and WING-readiness gates
closed. It reuses the canonical method and priority data from the backend rather
than maintaining a second method inventory.

## Rejected Alternatives

- Relying only on the large operator-review packet was rejected because it is
  detailed but not a compact goal-completion audit.
- Waiting for real WING proof was rejected for this slice because the user goal
  explicitly covers the safe offline proposal layer before live authority.
- Marking the project complete from green tests was rejected because
  `Docs/PROJECT_COMPLETION_GATE.md` requires real hardware, owner and quality
  evidence that is not present.

## Implementation Plan

- Add `tools/audit_correction_method_goal.py`.
- Add regression tests for the passing audit, JSON/Markdown writing,
  live-authority tamper rejection, and missing objective-family rejection.
- Embed `goal-audit.json` and `goal-audit.md` in the standard
  correction-method operator-review export bundle.
- Keep the existing operator-review export and no-WING gate as separate
  evidence surfaces.
- Record this ADR and operator-facing report.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/audit_correction_method_goal.py tools/export_correction_method_operator_review_packet.py tools/verify_correction_method_operator_review_export.py tests/test_processing_proposal_contracts.py`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_processing_proposal_contracts.py -q -k 'correction_method_goal_audit or operator_review_export_verifier_accepts_bundle or operator_review_export_writes_readonly_artifacts or tampered_goal_audit'`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_processing_proposal_contracts.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/audit_correction_method_goal.py --run-id 20260624T073100Z_correction_method_goal_audit --generated-at-utc 2026-06-24T07:31:00Z`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/export_correction_method_operator_review_packet.py --run-id 20260624T075458Z_correction_method_operator_review`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_correction_method_operator_review_export.py artifacts/correction_method_operator_review/20260624T075458Z_correction_method_operator_review --output artifacts/correction_method_operator_review/20260624T075458Z_correction_method_operator_review/verification-report.json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

The main risk is that a passed offline audit is misread as live readiness.
The audit and report mitigate this by emitting `contacts_wing: false`,
`live_authority_granted: false`, `readiness_closure_allowed: false`,
`live_apply_allowed: false`, `queue_import_allowed: false`, and
`hardware_proof_required_before_live: true`.

Rollback is safe: remove the audit tool, tests, generated audit artifacts,
embedded export entries, this ADR and the report. No runtime console behavior
or live-control path is widened by this decision.
