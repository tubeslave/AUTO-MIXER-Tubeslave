# Proposal: WING MVP Read-Only Artifact Contract Constants

## Short thesis

Centralize the WING read-only proof artifact identity in
`backend/wing_readonly_preflight_contract.py` so runner, packet, release-check,
evidence, and supervised-proof code cannot drift independently.

## What I understood

The current no-WING rails require a real `wing_readonly_preflight.v1`
`summary.json` plus `events.jsonl`. The same names appear in multiple tools and
tests. That duplication is a small but real safety risk because a future change
could update one surface and leave another accepting or requesting a different
artifact.

## Proposed solution

- Add shared constants for schema, summary artifact name, event log artifact
  name, proof summary artifact, and next manual artifact requirement.
- Use those constants in the read-only preflight runner, evidence intake,
  handoff packet builder, no-WING release check, and supervised proof plan.
- Update focused tests to assert those paths and artifact identities through the
  shared contract constants.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_evidence.py`
- `backend/supervised_wing_proof_plan.py`
- `tools/wing_readonly_preflight_runner.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- Focused tests for those tools.

## Alternatives considered

- Leave literals in place. Rejected because this lets packet and proof planning
  drift apart.
- Create a new schema module. Rejected because the existing preflight contract
  module is already the right ownership boundary.

## Risks

- Over-broad replacement could touch unrelated `summary.json` files. Avoided by
  changing only WING read-only preflight-specific code paths.
- Contract constants must preserve existing public values.

## Test plan

- Run focused preflight runner, handoff packet, no-WING release, evidence,
  review, supervised proof plan, Project Director, and guard tests.
- Run Project Director and direction guard command checks.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may prefer leaving stable literals alone. Centralizing them is worth it
because these strings define what real WING evidence can unlock later.
