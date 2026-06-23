# Proposal: WING MVP Supervised Prestart Artifact-Root Marker Gate

## Short Thesis

Promote the accepted-candidate artifact-root trace from an implementation detail
to an explicit supervised prestart gate marker.

## What I Understood

The current WING MVP is still blocked on a real read-only connection proof. Until
that proof exists, all work must improve the offline safety rail without implying
readiness. The accepted-candidate artifact-root contract is now enforced by code,
but operator-facing final gate wording must match that enforcement.

## Proposed Solution

- Add `valid accepted-candidate artifact-root trace` to the supervised prestart
  gate step and required marker list.
- Add `requires_accepted_candidate_artifact_root_trace: true` to the structured
  supervised prestart gate object.
- Update handoff/no-WING/Project Director tests and docs so the current first
  command covers the new baseline.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `.ai/project-layer/wing_mvp_blockers.md`

## Alternatives Considered

- Leave the marker implicit through proof-plan summary fields. Rejected because
  the final operator gate should be auditable without inferring hidden code
  behavior.

## Risks

- Exact string fixtures and baseline markers may drift. Mitigation: run focused
  handoff/no-WING/Project Director tests and full pytest.

## Test Plan

- Targeted handoff/no-WING/rails/Project Director tests.
- Focused WING/no-WING tests.
- Full standard pytest.
- no-WING quick, memory lint, Project Director, and `git diff --check`.

## Where The Other Agent May Disagree

Kimi may argue that artifact-root validity is already covered by required
summary fields. I would keep the explicit marker because this gate is read by
operators and release automation, not only by the proof-plan implementation.
