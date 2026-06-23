# Proposal: WING MVP Runbook Accepted-Decision Artifact-Root Shape

## Short Thesis

The operator-facing accepted-decision example should match the machine-enforced
review decision schema by including `artifact_root`.

## What I Understood

The current WING MVP cannot advance to supervised proof until a real read-only
capture is reviewed and accepted. The review-decision ledger now requires
accepted decisions to preserve `artifact_root`, but the runbook example did not
show that field.

## Proposed Solution

- Add `artifact_root` to the runbook minimum accepted-decision JSON.
- Add `artifact_root` to the list of fields that must match the captured files.
- Add `artifact_root` to the list of template-aligned fields after editing the
  pending decision JSON.
- Add a rails regression that checks those runbook strings.

## Likely Files To Touch

- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tests/test_wing_mvp_rails.py`

## Alternatives Considered

- Rely on generated pending templates only. Rejected because the runbook is the
  operator-facing source during hardware sessions.

## Risks

The change is low risk; the main failure mode is stale exact-string tests. The
runtime behavior does not change.

## Test Plan

- Targeted rails/review-decision tests.
- WING-focused tests.
- Full standard pytest.
- no-WING quick, memory lint, Project Director, and `git diff --check`.

## Where The Other Agent May Disagree

Kimi may say the generated template is enough. I would keep the runbook aligned
because manual hardware workflows depend on the visible example.
