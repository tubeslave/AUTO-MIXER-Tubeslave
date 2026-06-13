# Proposal: WING MVP Physical Readback Contract

## Short thesis

Centralize the physical readback status/source identity in the WING read-only
preflight contract so runner and evidence intake cannot disagree about what
counts as real proof.

## What I understood

The current rail is not live automation; it is proving a read-only WING
connection with real physical readbacks. If the runner accepts one set of
statuses/sources and evidence intake accepts another, readiness review can be
based on inconsistent proof semantics.

## Proposed solution

- Add shared physical readback status/source constants to
  `backend/wing_readonly_preflight_contract.py`.
- Use them in `tools/wing_readonly_preflight_runner.py`.
- Use them in `backend/wing_readonly_preflight_evidence.py`.
- Have no-WING compare runner and evidence physical-readback identities against
  the shared contract.
- Add summary fields and fail-closed violations for drift.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/no_wing_release_check.py`
- Focused tests for those surfaces.

## Alternatives considered

- Leave local sets duplicated. Rejected because these values define what real
  WING proof means.
- Only add direct unit tests. Rejected because no-WING should reject drift before
  operator hardware work.

## Risks

- Future backend response naming changes will need a deliberate contract update.
  That is desirable for a proof boundary.
- If no-WING cannot introspect the runner, it fails closed.

## Test plan

- Compile touched modules.
- Run focused runner, evidence, no-WING, director, guard, and rails tests.
- Run no-WING quick with no hardware and no backend writes.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may argue these names are stable enough to duplicate. The counterpoint is
that physical-readback identity is the core of the first blocker, so it belongs
in the shared preflight contract and release gate.
