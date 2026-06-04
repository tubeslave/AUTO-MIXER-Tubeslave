# ADR: WING Proof Artifact Schema

Date: 2026-06-04

## 1. Context

The supervised proof report, promotion evidence bridge, and WING proof plan now
form an evidence chain. That chain depends on physical proof artifacts being
unambiguous. Current historical WING operator proposal summaries can include
`execute=true` and readback fields without explicit physical proof metadata.

## 2. Options considered

- Infer physical proof from historical fields.
- Retroactively promote old artifacts.
- Add explicit proof schema fields to future runner summaries.

## 3. Decision

Add explicit proof metadata to `wing_operator_proposal_validation_runner.py`
summaries: schema id, runner id, `proof_mode`, `physical_wing`, and
`physical_proof_requires_execute`.

## 4. Why this won

It keeps promotion evidence current and auditable. Future physical proof runs
will be classifiable by the evaluator without guessing, while legacy artifacts
remain unpromoted unless rerun with the new schema.

## 5. Rejected alternatives

- Inference from `execute=true` was rejected because it could over-trust old or
  incomplete artifacts.
- Retrofitting old artifacts was rejected because promotion should be backed by
  explicit current proof.
- Keeping the old schema was rejected because valid physical proof could be
  invisible to the report layer.

## 6. Implementation plan

1. Add proof metadata fields to runner summary initialization.
2. Test dry-run planning summaries.
3. Test executed summaries.
4. Test evaluator acceptance of explicit physical schema with physical sources.

## 7. Test plan

- Compile touched Python files.
- Run runner/evaluator/report/plan focused tests.
- Run broader readiness and no-WING boundary tests.
- Run the standard full pytest command.

## 8. Risks and rollback

Risk: a fake test backend writes `proof_mode=physical`. Promotion still requires
physical readback sources and manual safety review. Rollback is to remove the
metadata fields; no live write path is introduced.
