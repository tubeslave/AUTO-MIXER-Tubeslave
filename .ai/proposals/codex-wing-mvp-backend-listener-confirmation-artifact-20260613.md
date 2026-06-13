# Codex Proposal: WING MVP Backend Listener Confirmation Artifact

## Short thesis

Add a strict operator confirmation JSON for busy backend-port prestart checks so
an already-running intended backend can be confirmed without letting Codex stop
or reconfigure local processes.

## What I understood

The current blocker is not WING hardware behavior yet; it is the local prestart
gate. A listener on port `8765` might be the intended backend, but Codex must
not assume that. The missing piece is a durable operator artifact that can turn
manual confirmation into a validated input.

## Proposed solution

- Add `--backend-listener-confirmation` to `tools/no_wing_release_check.py`.
- Accept only `wing_backend_port_listener_confirmation.v1` JSON with an
  accepted status, matching host/port/PID/command line, and no live authority.
- Keep no-file and mismatched-file paths fail-closed.
- Render accepted confirmation details in `summary.md`.
- Update the WING read-only runbook with the JSON shape.
- Add focused tests and report the decision.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`
- `Docs/reports/...`
- `Docs/adr/...`
- `Docs/INDEX.md`

## Alternatives considered

- Keep the gate purely fail-closed and ask the operator to clear the port.
- Auto-detect that the listener is acceptable from process command line alone.
- Stop or reconfigure the listener from the gate.

## Risks

The confirmation artifact could be stale. To reduce that risk, validation must
match the current listener PID and command line on every run.

## Test plan

- Focused no-WING, WING rails, Project Director, and Direction Guard tests.
- Project Director and Direction Guard JSON checks.
- `git diff --check`.
- `tools/memory/memory_lint.py`.
- Full backend pytest.

## Where the other agent may disagree

Kimi may prefer always requiring a free port. I would keep this confirmation
path because the intended backend is supposed to exist before WING read-only
preflight, and the artifact keeps the decision explicit, reviewable, and safe.
