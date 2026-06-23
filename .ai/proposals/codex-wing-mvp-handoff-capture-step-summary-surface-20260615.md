# Codex Proposal: WING MVP Handoff Capture Step Summary Surface

## Short Thesis

Make the no-WING summary explicitly prove that backend/WING connection
confirmation precedes the read-only capture step and that the capture step still
matches the shared handoff contract.

## What I Understood

The first MVP blocker remains `WING read-only connection proof`. Without real
hardware, progress should harden the operator handoff and fail-closed checks,
not claim readiness or live authority.

## Proposed Solution

- Add capture-step expected index/text and match verdicts to handoff artifact
  validation detail.
- Promote backend-confirmation and capture-step fields into the flat no-WING
  summary surface.
- Render those fields in `summary.md`.
- Add tests, Project Director baseline markers, and documentation.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`

## Alternatives Considered

- Keep the fields nested in the handoff artifact. Rejected because the no-WING
  summary is the operator's pre-WebSocket gate.
- Rely only on the operator sequence hash. Rejected because explicit capture
  fields make stale or reordered handoffs easier to diagnose.

## Risks

- More summary fields increase noise. Mitigation: keep them grouped with the
  existing operator-sequence gate fields and preserve fail-closed live-authority
  denial.

## Test Plan

- Compile touched Python files.
- Run focused no-WING and handoff packet tests.
- Run no-WING quick validation.
- Run Project Director selected-action check.
- Run full pytest.
- Run memory lint and diff whitespace check.

## Where The Other Agent May Disagree

Kimi may prefer only a compact structured operator-sequence object. The flat
surface is more consistent with the current Project Director marker checks.
