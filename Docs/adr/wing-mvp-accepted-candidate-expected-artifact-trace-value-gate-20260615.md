# ADR: WING MVP Accepted Candidate Expected Artifact Trace Value Gate

## 1. Context

The WING supervised proof path requires an accepted read-only preflight review
decision before any physical proof. The proof-plan summary already carries
accepted-candidate expected artifact trace fields, but those fields must also
match the canonical trace from the review decision ledger.

## 2. Options considered

- Keep the required-field presence gate only.
- Check only the trace completion boolean.
- Strictly compare every source-id keyed trace field against the canonical
  trace in no-WING validation.

## 3. Decision

Add value comparisons in `validate_supervised_proof_report_next_wing_plan()` for
trace source ids, expected artifact lists, expected summary/events artifact
maps, summary/events path match flags, missing source ids, and trace completion.

## 4. Why this won

The no-WING release check is the operator-facing fail-closed gate before a WING
session. Strict value comparisons prevent a proof plan from passing with all
required keys present but stale or incorrect accepted-candidate artifact
bindings.

## 5. Rejected alternatives

Required-field presence only was rejected because it cannot detect stale values.
Completion-only checks were rejected because they do not prove which accepted
source id is bound to which expected artifacts.

## 6. Implementation plan

- Add exact normalizers for trace maps in the release-check tool.
- Compare reported proof-plan summary values against the canonical trace.
- Add a parametrized regression test that mutates each trace field.
- Record the validation in the WING MVP report and progress surfaces.

## 7. Test plan

- Targeted no-WING trace mismatch regression test.
- Existing next proof-plan summary surface test.
- no-WING quick artifact check.
- Project Director check.
- Focused WING/no-WING suite.
- Full pytest, memory lint, and whitespace diff check.

## 8. Risks and rollback

Risk: generated reports with copied or stale trace maps now fail no-WING
validation. Rollback is to remove the value comparisons while keeping the
required trace fields in the proof-plan summary.
