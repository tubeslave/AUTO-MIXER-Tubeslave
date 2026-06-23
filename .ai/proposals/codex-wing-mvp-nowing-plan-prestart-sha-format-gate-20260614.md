# Codex Proposal: WING MVP no-WING Plan Prestart SHA Format Gate

## Short thesis

The no-WING offline plan should reject malformed prestart summary digests before
they can be treated as handoff packet plan truth.

## What I understood

The active blocker remains `WING read-only connection proof`. The accepted
candidate helper now requires a 64-character lowercase SHA-256 digest, but the
plan artifact validator still allowed any non-empty
`backend_port_prestart_summary_sha256`. A handoff packet could then match that
same malformed plan value and only appear consistent.

## Proposed solution

Expose the SHA-256 hex helper from the shared preflight contract and use it in
`tools/no_wing_release_check.py` when validating WING read-only plan artifacts.
Add regression coverage for both the plan artifact itself and a handoff packet
whose packet SHA matches a malformed plan SHA.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- Council/report docs for this slice.

## Alternatives considered

- Rely only on handoff packet SHA-vs-plan equality. Rejected because equality
  does not prove digest validity.
- Keep the helper private and duplicate logic in no-WING. Rejected to keep the
  accepted-candidate and plan artifact gates on the same digest rule.

## Risks

Old malformed local plan fixtures will fail. That is intended because such
fixtures do not prove a bound no-WING prestart summary.

## Test plan

- Focused no-WING plan/handoff SHA tests.
- Full `tests/test_no_wing_release_check.py`.
- Project Director rails, memory lint, and `git diff --check`.

## Where the other agent may disagree

Kimi may prefer accepting uppercase hex. The current repo-generated digests
come from `hashlib.hexdigest()` and are lowercase, so strict lowercase keeps
operator evidence unambiguous.
