# ADR: WING MVP Decision Review Artifact Root Contract Surface

## 1. Context

The accepted WING read-only review-decision ledger computes accepted-candidate
prestart and artifact-root contract fields. no-WING summaries also expose those
fields. The standalone decision-review operator report still decided
`accepted_decision_ready` from prestart contract fields only, so the direct
operator report could lag behind the stricter ledger/no-WING contract.

## 2. Options considered

- Leave the standalone report prestart-only.
- Recompute artifact-root validity inside the tool.
- Propagate and enforce the ledger artifact-root fields in the report.

## 3. Decision

Propagate accepted-candidate artifact-root contract fields from the ledger into
the standalone decision-review report, and require them before reporting
`accepted_decision_ready`.

## 4. Why this won

The decision-review report now matches the ledger and no-WING gate while
keeping a single source of truth for artifact-root validation.

## 5. Rejected alternatives

Prestart-only readiness leaves a stale operator surface. Recomputing inside the
tool duplicates the shared ledger contract and risks drift.

## 6. Implementation plan

- Import the artifact-root blocker constant.
- Read artifact-root contract fields from the ledger summary.
- Require artifact-root contract validity for report ready status.
- Render artifact-root fields and fallback blocker in markdown.
- Add valid and invalid artifact-root contract regressions.

## 7. Test plan

Run targeted decision-review tool tests, focused WING/no-WING tests,
WING-focused tests, standard pytest, no-WING release check, diff check, memory
lint, and Project Director.

## 8. Risks and rollback

Risk is limited to stricter offline decision-review reports. Rollback is
reverting the report propagation, ready-status check, markdown lines, and tests.
