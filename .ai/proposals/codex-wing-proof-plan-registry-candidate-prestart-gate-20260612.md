# Codex Proposal: WING Proof Plan Registry Candidate Prestart Gate

## Short thesis

The processing registry should identify `supervised_wing_proof_plan` as gated by accepted-candidate prestart trace, not merely as a generic operator checklist.

## What I understood

The code, release gate, and runbook now enforce the candidate-prestart boundary. The registry is the machine-readable inventory used to explain method status and should expose the same blocker.

## Proposed solution

- Update the registry summary for `supervised_wing_proof_plan`.
- Set its `blocked_reason` to the candidate-prestart blocker.
- Add a `candidate_prestart_trace_required` note.
- Update the UI next-step wording for the method.
- Extend the existing registry test to assert the blocker, summary wording, note, and operator approval boundary.

## Likely files to touch

- `backend/processing_registry.py`
- `tests/test_processing_registry.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`

## Alternatives considered

- Leave registry wording unchanged. That makes the inventory weaker than the actual proof-plan gate.
- Add a new method id. Not needed; this is a metadata alignment for the existing method.

## Risks

- Minimal. The method remains shadow-only and non-executing.

## Test plan

- Focused processing registry pytest.
- Affected WING/no-WING tests.
- Full standard pytest suite.
- Diff and memory lint.

## Where the other agent may disagree

Kimi may prefer a broader registry audit for all WING proof methods. I would keep this scoped to the method whose blocker changed and leave broader inventory grooming for a separate pass.
