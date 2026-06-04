# Codex Proposal: Supervised Proof Remediation Plan

Task id: supervised-proof-remediation-plan-20260604
Date: 2026-06-04

## 1. Short thesis

Blocked proof artifacts should produce a tested remediation checklist, not only
a blocked status. The checklist must remain read-only and operator-executed.

## 2. What I understood

The project needs a credible path from dry-run and historical supervised proof
toward physical promotion evidence. The latest fail-closed legacy classifier
correctly blocks old executed WING summaries without schema, but it leaves the
next action implicit.

## 3. Proposed solution

Add `supervised_proof_remediation_plan`, a pure report transformer that:

- summarizes blocker and value-type counts;
- creates rerun actions for legacy ambiguous WING artifacts;
- creates cleanup actions for invalid summaries;
- creates manual review actions for other blocked proof rows;
- embeds the plan in `tools/evaluate_supervised_proof_artifacts.py` output.

## 4. Likely files to touch

- `backend/supervised_proof_remediation_plan.py`
- `tools/evaluate_supervised_proof_artifacts.py`
- `backend/processing_registry.py`
- focused tests and ADR/brief artifacts

## 5. Alternatives considered

- Leave blocked rows as a raw list.
- Fold remediation into the WING proof plan.
- Execute remediation reruns automatically.

## 6. Risks

The generated rerun command contains `--execute`, so the report must clearly
mark `execution_allowed_by_plan=false`, `live_write=false`, and
`operator_required=true`.

## 7. Test plan

- Unit-test legacy rerun action shape and no execution authority.
- Unit-test invalid-summary cleanup actions.
- Assert report embedding and registry shadow-only status.
- Run focused, broad, smoke, and full pytest validation.

## 8. Where the other agent may disagree

Kimi may argue that the WING proof plan already covers missing physical
evidence. The remediation layer is narrower: it names concrete blocked
artifacts and explains how to replace or clean them up.
