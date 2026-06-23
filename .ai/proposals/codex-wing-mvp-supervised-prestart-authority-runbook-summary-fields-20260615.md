# Codex Proposal: WING MVP Supervised Prestart Authority Runbook Summary Fields

## Short Thesis

Publish the supervised prestart gate authority denial, required decision status,
and target runbook as flat no-WING summary fields.

## What I Understood

The current WING rail is still offline/no-authority. Operators must be able to
verify from summary surfaces that the supervised prestart gate does not grant
live authority and points only to the supervised proof runbook after manual
acceptance.

## Proposed Solution

- Derive the flat fields from `supervised_prestart_gate`.
- Render them in `summary.json` and operator-readable `summary.md`.
- Add focused tests and Project Director baseline markers.
- Document the change as a summary-surface hardening only.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`

## Alternatives Considered

- Keep the fields nested only. Rejected because the safety gate should be
  grep-friendly from release summaries.
- Move authority fields into a broader registry. Deferred to avoid unrelated
  refactor risk.

## Risks

- Flat fields could drift from the structured gate. Mitigation: derive from the
  structured gate and assert top-level/Markdown parity.

## Test Plan

- Compile touched Python files.
- Run focused no-WING and Project Director tests.
- Run no-WING quick validation.
- Run the full pytest suite.
- Run memory lint and diff whitespace check.

## Where The Other Agent May Disagree

Kimi may prefer a generated projection table. The narrower implementation is
lower risk for the active WING read-only proof rail.
