# Codex Proposal: Legacy Proof Artifact Classification

Task id: legacy-proof-artifact-classification-20260604
Date: 2026-06-04

## 1. Short thesis

Legacy executed WING summaries without explicit proof schema must fail closed.
They can remain useful audit records, but they should not be reported as
`dry_run_ready` or promotion evidence.

## 2. What I understood

The previous schema update made future WING proof summaries explicit. The weak
spot is historical `wing_operator_proposal_*` summaries that have `execute=true`
but no `proof_mode` or `physical_wing`. Treating them as dry-run artifacts
overstates readiness.

## 3. Proposed solution

Add a `legacy_ambiguous` proof mode inferred only for old executed
`wing_operator_proposal` payloads without explicit mode fields. The evaluator
adds `legacy_physical_schema_missing` as a common blocker, so these rows become
`blocked` even when their readback fields look complete.

## 4. Likely files to touch

- `backend/supervised_proof_artifact_evaluator.py`
- `tests/test_supervised_proof_artifact_evaluator.py`
- `tests/test_supervised_proof_artifact_report.py`
- `.ai/briefs/legacy-proof-artifact-classification-20260604.md`
- `Docs/adr/legacy-proof-artifact-classification-20260604.md`

## 5. Alternatives considered

- Infer physical proof from `execute=true`.
- Keep classifying missing-mode artifacts as dry-run.
- Ignore all historical WING summaries.

## 6. Risks

Some old physical proof rows will move from dry-run-ready to blocked. That is
intentional: historical rows can inform manual audit, but promotion should use
current explicit schema.

## 7. Test plan

- Unit test legacy executed WING summary normalization and blockers.
- Report test that legacy rows increment blocked counts, not dry-run counts.
- Focused, broad, smoke, and full pytest validation.

## 8. Where the other agent may disagree

Kimi may prefer accepting old physical readback as supervised evidence. The
safer counterargument is that promotion evidence should not depend on inference
from pre-schema artifacts.
