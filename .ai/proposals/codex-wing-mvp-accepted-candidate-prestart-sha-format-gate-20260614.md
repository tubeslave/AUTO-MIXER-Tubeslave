# Codex Proposal: WING MVP Accepted-Candidate Prestart SHA Format Gate

## Short thesis

An accepted WING read-only candidate should not become proof-plan-ready with an
arbitrary non-empty prestart digest string.

## What I understood

The active blocker remains `WING read-only connection proof`. Current offline
rails require accepted candidates to carry a no-WING prestart summary path and
SHA256, but the downstream shared helper only checked that the SHA field was
non-empty.

## Proposed solution

Add a small SHA-256 hex validator to the shared prestart contract helper and
return `backend_port_prestart_summary_sha256_invalid` for malformed digests.
Cover the helper directly, the review-decision ledger, and the supervised proof
plan so malformed digests keep `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`
in force.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `tests/test_supervised_wing_proof_plan.py`
- Council/report docs for this slice.

## Alternatives considered

- Rely only on evidence intake file hashing. Rejected because downstream rows
  must stay fail-closed if malformed data is passed in from a ledger or fixture.
- Accept uppercase hex. Rejected for now because repo-produced SHA256 values
  come from `hashlib.hexdigest()` and are lowercase; a stricter contract reduces
  ambiguous operator evidence.

## Risks

The stricter helper can invalidate old local fixtures that used placeholder
non-hex strings. Current tests use valid lowercase 64-character fixtures.

## Test plan

- Focused review-decision tests.
- Focused supervised proof-plan tests.
- Compile, memory lint, Project Director guard, and diff check.

## Where the other agent may disagree

Kimi may prefer a reusable digest helper shared with all no-WING validators.
That can be extracted later if a second digest format gate appears; this patch
keeps the current safety change scoped to accepted-candidate prestart traces.
