# Independent Proposal — M115 WING Firmware Release Suffix

## 1. Short thesis

Use one shared full-match predicate for both backend issuance and runner
verification. Preserve the exact raw firmware string in authority and artifact.

## 2. What I understood

The current regex is duplicated in `backend/server.py` and
`tools/wing_attended_soundcheck_runner.py`. Updating only one creates a safe but
permanent backend/runner contract mismatch.

## 3. Proposed solution

Place a compiled predicate in `backend/master_capture_route_proof.py`, already
inside the critical source set and imported by both callers. Accept
`3.0.5(?:build-suffix)?` with only an optional exact terminal `:release` tag.
Do not normalize or truncate the raw firmware value.

## 4. Likely files to touch

- `backend/master_capture_route_proof.py`
- `backend/server.py`
- `tools/wing_attended_soundcheck_runner.py`
- their focused tests

## 5. Alternatives considered

Duplicating the revised regex is smaller in line count but risks future drift.
Adding colon to the generic character class or using `startswith` is too broad.

## 6. Risks

Changing only the issuance gate would still fail HW01 at runner semantic
verification. Normalizing the version would weaken exact evidence equality.

## 7. Test plan

Test base/build/real release forms and reject wrong base, direct tag, debug tag,
extra tag and slash separator. Prove server issuance and runner acceptance keep
the exact full string and zero write count.

## 8. Where the other agent may disagree

Codex may prefer the smallest inline regex edit. The shared predicate costs a
few lines but removes a demonstrated two-gate drift hazard.

