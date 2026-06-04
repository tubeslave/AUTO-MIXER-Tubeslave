# Codex Proposal: WING Proof Artifact Schema

## 1. Short thesis

Make physical WING proof artifacts self-identifying so the proof report can
classify future evidence without guessing from runner names or status fields.

## 2. What I understood

The current proof report found no physical promotion candidates. Inspection
shows a schema problem: `wing_operator_proposal_validation_runner.py` writes
`execute=true` and readback evidence, but no explicit `proof_mode` or
`physical_wing`. Future proof runs should not depend on inference.

## 3. Proposed solution

Add explicit fields to every runner summary:

- `proof_artifact_schema`;
- `proof_runner`;
- `proof_mode`;
- `physical_wing`;
- `physical_proof_requires_execute`.

Dry-run planning summaries use `proof_mode=dry_run` and `physical_wing=false`.
Executed physical proof summaries use `proof_mode=physical` and
`physical_wing=true`.

## 4. Likely files to touch

- `tools/wing_operator_proposal_validation_runner.py`
- `tests/test_wing_operator_proposal_validation_runner.py`
- `tests/test_supervised_proof_artifact_evaluator.py`
- `.ai/briefs/wing-proof-artifact-schema-20260604.md`
- `Docs/adr/wing-proof-artifact-schema-20260604.md`

## 5. Alternatives considered

- Infer physical proof from `execute=true`: rejected because legacy artifacts
  can be incomplete or ambiguous.
- Mark old artifacts as physical retroactively: rejected because promotion
  should use explicit current evidence.
- Leave classification as dry-run: rejected because future valid physical proof
  would remain invisible to the report.

## 6. Risks

- A test fake backend can produce a `physical` schema without real hardware.
  The evaluator still requires physical readback sources for promotion.
- Legacy artifacts remain unpromoted. That is intentional until a new proof run
  writes explicit schema fields.

## 7. Test plan

- Assert dry-run planning summaries are explicit dry-run artifacts.
- Assert executed summaries are explicit physical artifacts.
- Assert evaluator accepts the explicit physical schema only when readback
  sources are physical.
- Run focused and full repository tests.

## 8. Where the other agent may disagree

Kimi may prefer inference from existing artifacts. The safer path is not to
promote ambiguous historical artifacts; rerun the bounded proof with explicit
schema when the operator is ready.
