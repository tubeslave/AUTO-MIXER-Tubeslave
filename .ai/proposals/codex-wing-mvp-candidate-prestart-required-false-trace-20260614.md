# Proposal: WING MVP candidate prestart required-false trace

## Short thesis

Propagate the no-WING prestart required-false detail fields into the
accepted-candidate review contract so reviewer-visible traces cannot drop safety
denial fields after evidence intake.

## What I understood

The first open MVP blocker is still `WING read-only connection proof`. Offline
tooling must make the future real artifact review stricter without granting live
authority.

## Proposed solution

- Extend `WingReadonlyPreflightArtifactRecord` with the required-false prestart
  detail fields.
- Include those fields in artifact review and decision candidate review rows.
- Make `wing_readonly_preflight_candidate_prestart_contract_violations` reject
  missing or truthy fields.
- Render the fields in artifact and decision review markdown.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_evidence.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tools/review_wing_readonly_preflight_decisions.py`
- WING read-only focused tests and docs.

## Alternatives considered

Keep the check only in evidence intake. Rejected because the proof-plan gate
uses candidate review rows, so dropping fields there weakens reviewer evidence.

## Risks

Older synthetic candidate rows without these fields will now fail the
accepted-candidate prestart contract, which is intentional for supervised proof
planning.

## Test plan

- Focused evidence tests.
- Artifact review tests.
- Decision ledger and decision report tests.
- Project Director/WING rail tests.
- Full standard pytest if the focused stack is clean.

## Where the other agent may disagree

Kimi may argue the helper already had enough path/SHA/no-contact coverage. The
counterpoint is that the current WING blocker text now treats these false
fields as part of the trace reviewers must see.
