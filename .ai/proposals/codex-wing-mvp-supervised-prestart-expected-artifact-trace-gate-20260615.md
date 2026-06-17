# Codex Proposal: WING MVP Supervised Prestart Expected Artifact Trace Gate

## Short thesis

The supervised prestart gate should encode the expected-artifact trace
requirement as structured data, not only as operator text.

## What I understood

The no-WING handoff currently says the final gate needs an
expected-artifact-bound accepted source id, while the shared
`supervised_prestart_gate` object only requires prestart and artifact-root
traces. That leaves a narrow drift path where text and nested proof-plan
summary fields carry the requirement but the canonical gate object does not.

## Proposed solution

Add `requires_accepted_candidate_expected_artifact_trace: true` to
`expected_supervised_prestart_gate()`, then require the same marker in Current
First Command, Project Director baseline coverage, autoloop prompt/README, and
focused tests.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/update_mvp_progress.py`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- focused no-WING, handoff-packet, and Project Director tests

## Alternatives considered

- Leave the requirement only in text and proof-plan summary fields.
- Add a separate new gate object.

## Risks

Existing fixtures with hard-coded expected gate objects must be updated.
Runtime risk is low because this remains offline validation metadata and grants
no live authority.

## Test plan

Run focused tests for no-WING release check, handoff packet, Project Director,
and the supervised proof plan; then run no-WING quick and full pytest if green.

## Where the other agent may disagree

Kimi may prefer keeping the final gate object smaller and using only
`required_supervised_proof_plan_summary_fields`. The counterargument is that
the structured gate is the operator-visible unlock contract and should name all
three trace prerequisites explicitly.
