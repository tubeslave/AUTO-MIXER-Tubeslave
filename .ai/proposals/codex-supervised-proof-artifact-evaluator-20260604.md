# Codex Proposal: Supervised Proof Artifact Evaluator

## 1. Short thesis

Add a transport-free proof artifact evaluator so supervised promotion can be
based on concrete evidence instead of broad capability claims.

## 2. What I understood

The repository has a narrow supervised fader/gain path and dry-run/physical
proof runners. The system needs a reusable gate that reads their summaries and
reports whether an artifact is dry-run-ready, a physical supervised candidate,
or blocked. This must not widen live authority.

## 3. Proposed solution

Create `backend/supervised_proof_artifact_evaluator.py` with:

- normalized input and evaluation dataclasses;
- validation for pre-read, apply readback, rollback readback, rollback proof,
  emergency-stop proof, blocked follow-up proof, operator approval, and
  idempotency;
- physical-source checks when promotion requires physical proof;
- parser support for runner-style summary dictionaries;
- snapshot output for reports and registry surfaces.

## 4. Likely files to touch

- `backend/supervised_proof_artifact_evaluator.py`
- `backend/processing_registry.py`
- `tests/test_supervised_proof_artifact_evaluator.py`
- `tests/test_processing_registry.py`
- `.ai/briefs/supervised-proof-artifact-evaluator-20260604.md`
- `Docs/adr/supervised-proof-artifact-evaluator-20260604.md`

## 5. Alternatives considered

- Promote gain after dry-run proof: rejected because physical WING proof is
  still missing.
- Put the checks inside the runners: useful locally, but registry/reporting
  surfaces need a reusable read-only evaluator.
- Treat existing safety rows as enough: rejected because they do not parse
  concrete proof artifacts.

## 6. Risks

- A dry-run artifact could be mistaken for physical proof. The evaluator keeps
  dry-run status separate and blocks promotion when physical proof is required.
- Summary formats may drift. The parser accepts both normalized input and
  runner-style dictionaries, while missing fields fail closed.
- Physical proof could be incomplete. Physical promotion requires physical
  pre-read, apply readback, and rollback readback sources.

## 7. Test plan

- Verify the evaluator has no mixer transport or direct write tokens.
- Verify complete dry-run proof is not promotable when physical proof is
  required.
- Verify complete physical fader proof is supervised-candidate only.
- Verify missing pre-read, rollback, emergency-stop, unsupported families, and
  gain-up without reason block.
- Verify snapshot counts and autonomous-ready ids stay empty.

## 8. Where the other agent may disagree

Kimi may prefer to run a real gain proof next. That is still the right later
step, but this evaluator makes the evidence contract explicit before any
promotion claim.
