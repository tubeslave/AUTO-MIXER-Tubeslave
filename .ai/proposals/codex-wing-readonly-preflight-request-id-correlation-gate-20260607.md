# Proposal: WING Read-Only Preflight Request-ID Correlation Gate - 2026-06-07

## 1. Short thesis

Make every read-only WING parameter proof correlate request and response with a
stable `request_id`, then fail closed when that correlation is missing.

## 2. What I understood

The project is not ready to move past the first WING blocker until a real
read-only preflight artifact exists and is manually reviewed. The capture tool
must therefore produce evidence strong enough to prove that each physical
readback belongs to the exact query sent by the runner.

## 3. Proposed solution

- Add deterministic request ids for runner-generated `query_mixer_parameter`
  requests.
- Echo `request_id` from `backend/handlers/mixer_handlers.py` for
  `mixer_parameter_value` and `mixer_parameter_read_blocked`.
- Block the runner when a response request id is missing or mismatched.
- Require the request-id policy in offline evidence and no-WING plan checks.
- Extend event-log validation to compare query and response request ids.

## 4. Likely files to touch

- `backend/handlers/mixer_handlers.py`
- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- focused WING preflight tests
- AI Council brief/proposal/ADR/report and `Docs/INDEX.md`

## 5. Alternatives considered

- Rely on event order only. Rejected because order proves sequence but not
  explicit request/response correlation.
- Use random UUIDs. Rejected because deterministic ids are simpler to review
  offline and easier to assert in plan artifacts.
- Add correlation only to summary artifacts. Rejected because the backend
  response itself must carry the proof field.

## 6. Risks

Older captured artifacts without request ids will no longer satisfy the updated
proof contract. That is acceptable because the first WING proof has not been
closed and should use the stronger contract.

## 7. Test plan

- Compile changed Python modules.
- Run WING read-only runner/evidence/review/no-WING focused tests.
- Run server handler tests for successful and blocked read-only parameter
  responses.
- Run no-WING quick and full pytest before commit.

## 8. Where the other agent may disagree

Kimi may prefer random per-run UUIDs. I would keep deterministic request ids
for this first hardware proof because the query set is small, ordered, and
operator-reviewed offline.
