# Proposal: WING MVP Decision Review Artifact Root Contract Surface

## 1. Short thesis

Align the standalone decision-review report with the stricter ledger contract
so an accepted decision cannot appear ready while its artifact-root trace is
invalid.

## 2. What I understood

The active MVP remains supervised WING Rack proof. The first blocker is still
real read-only WING evidence and operator review, but the offline tooling must
fail closed before that physical step is attempted.

## 3. Proposed solution

Read accepted-candidate artifact-root contract fields from the ledger, require
them for `accepted_decision_ready`, include them in JSON/markdown output, and
render the artifact-root blocker when the ledger summary is stale.

## 4. Likely files to touch

- `tools/review_wing_readonly_preflight_decisions.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`
- Council ADR/report/index files

## 5. Alternatives considered

- Rely on no-WING summary only.
- Keep the standalone report prestart-only.
- Recompute artifact-root contract independently in the tool.

## 6. Risks

The tool becomes stricter. The practical risk is an earlier blocked operator
report when artifact-root metadata is stale, which is the intended fail-closed
behavior.

## 7. Test plan

Run targeted decision-review tool tests, focused WING/no-WING tests,
WING-focused tests, standard pytest, no-WING release check, diff check, memory
lint, and Project Director.

## 8. Where the other agent may disagree

Kimi may prefer recomputing the artifact-root contract inside the tool. Reusing
the ledger fields keeps one source of truth and matches the existing prestart
contract flow.
