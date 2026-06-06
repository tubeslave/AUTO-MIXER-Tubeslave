# ADR: Supervised WING Proof Read-Only Preflight Gate

## Context

The supervised WING Rack proof order is read-only preflight, physical readback,
dry-run recommendation, explicit operator approval, bounded supervised apply,
readback, rollback, and artifact recording. `supervised_wing_proof_plan` could
previously emit physical fader/gain proof steps even when read-only preflight
evidence was missing.

## Options Considered

1. Keep the plan unchanged and document the order externally.
2. Gate proof plan steps on the autonomy readiness read-only preflight overlay.
3. Gate only inside the physical proof runner.

## Decision

Use option 2.

## Why This Won

The proof plan is the operator-facing and release-checked next-step artifact.
It should not advertise physical proof-runner commands before the read-only
connection/readback candidate exists.

## Rejected Alternatives

- External documentation alone is too easy to miss when an automated report
  includes a runnable command.
- Runner-only enforcement is necessary later but insufficient for readiness
  reports; the report itself must show the correct next blocker.

## Implementation Plan

- Add `blocked_by_readonly_preflight` to the proof plan statuses.
- Read the `wing_readonly_preflight_evidence` overlay from autonomy readiness.
- Emit zero physical proof steps until the preflight candidate is clean.
- Emit a separate non-executing read-only preflight checklist step while the
  plan is blocked by missing preflight evidence.
- Embed the readiness snapshot in the proof artifact report and reuse it for
  the next WING proof plan.
- Extend no-WING release validation and summaries with preflight fields.

## Test Plan

- `python3 -m py_compile backend/supervised_wing_proof_plan.py backend/autonomy_promotion_readiness.py backend/wing_readonly_preflight_evidence.py backend/processing_registry.py tools/evaluate_supervised_proof_artifacts.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_supervised_proof_artifact_report.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Risk: operators see a blocked plan until read-only preflight evidence is
captured. That is intentional and matches the WING MVP order. Rollback is a
normal git revert, with no live mixer state involved.
