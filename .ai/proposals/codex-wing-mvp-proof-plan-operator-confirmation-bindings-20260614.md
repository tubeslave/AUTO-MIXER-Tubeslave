# Codex Proposal: WING MVP Proof Plan Operator Confirmation Bindings

## Short thesis

Make the supervised WING proof plan fallback read-only preflight command match
the runner/runbook operator-confirmation contract, then make the no-WING gate
fail closed on stale weaker commands.

## What I understood

The current MVP still waits on real WING read-only connection proof. The
project can continue improving the offline rail only when the work directly
supports that selected blocker and does not claim readiness closure.

## Proposed solution

- Reuse canonical read-only preflight constants for fallback command channels
  and value types.
- Require `--operator-confirm-channels` to match `--channels`.
- Require `--operator-confirm-session-ready`.
- Preserve `--prestart-summary` and `--expected-plan` bindings.
- Extend `tools/no_wing_release_check.py` detail, JSON summary, and Markdown
  summary with these bindings.
- Add focused tests proving the old command shape would fail.

## Likely files to touch

- `backend/supervised_wing_proof_plan.py`
- `tools/no_wing_release_check.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`

## Alternatives considered

- Leave fallback command unchanged: rejected because it leaves a weaker
  operator path than the runner accepts for physical readback.
- Introduce a shared command builder: deferred because the local diff is enough
  and avoids a wider refactor.

## Risks

The summary schema grows by four fields. Existing stale reports without those
fields should fail the release gate, which is the desired behavior for this
contract.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/supervised_wing_proof_plan.py tools/no_wing_release_check.py`
- Focused WING rail pytest.
- Full standard pytest.

## Where the other agent may disagree

Kimi may prefer a shared command builder to remove duplication across runner,
handoff packet, and proof plan. That can be worthwhile later, but this task is
safer as a narrow contract alignment.
