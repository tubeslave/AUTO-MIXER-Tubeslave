# Codex Proposal: WING MVP No-WING Quick Busy-Port Fail-Closed

## Short thesis

Keep the current no-WING quick gate fail-closed artifact visible to Project
Director so the next WING preflight step starts with the real local blocker:
manual confirmation of the listener on backend port `8765`.

## What I understood

The first blocker still requires real WING read-only proof. A no-WING quick run
is allowed as a local prestart gate, and this run correctly failed closed before
any WING contact because a local Python listener is already bound to the backend
port.

## Proposed solution

- Commit the no-WING quick `summary.json` and `summary.md` artifact.
- Add a concise report documenting the busy-port fail-closed result and
  operator next step.
- Add the report to the first blocker evidence cell.
- Assert WING MVP rails and Project Director expose the report.

## Likely files to touch

- `artifacts/no_wing_release_check/20260613T090119Z_no_wing_release_check/`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`
- `Docs/reports/...`
- `Docs/adr/...`
- `Docs/INDEX.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

- Leave the artifact untracked and only mention the result in chat.
- Stop or reconfigure the listener automatically.
- Treat the fail-closed result as a readiness failure.

## Risks

The artifact contains local process metadata. It does not include secrets and is
operator-relevant, but it must not be mistaken for WING read-only proof.

## Test plan

- Focused WING rails, Project Director, and Direction Guard tests.
- Project Director JSON check.
- Direction Guard JSON check.
- `git diff --check`.
- `tools/memory/memory_lint.py`.
- Full backend pytest.

## Where the other agent may disagree

Kimi may prefer not committing runtime artifacts. I would commit this one
because `artifacts/no_wing_release_check/` is already used for tracked release
evidence, and the artifact is the current proof that the prestart gate is
blocking safely.
