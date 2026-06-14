# Codex Proposal: WING MVP no-WING Frontend Artifact Preservation

## Short thesis

Make no-WING frontend validation leave the checkout clean by preserving tracked
Playwright state around the frontend e2e command.

## What I understood

The active supervised WING MVP still waits for real WING read-only proof. The
useful offline improvement here is to keep validation cycles clean and
repeatable, not to claim readiness. A no-WING check that dirties tracked
frontend test metadata creates noise for Goal-mode handoffs and commit
preparation.

## Proposed solution

Add a `preserve_paths` field to `CommandCheck`. `run_command_check` snapshots
those repo-local files before the subprocess, restores existing files after the
subprocess, and removes a preserved path if the command created it from an
absent starting state. The frontend e2e check declares
`frontend/test-results/.last-run.json` as preserved.

Also extend repo-local command validation so preserved paths cannot point
outside the checkout.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-nowing-frontend-artifact-preservation-20260614.md`
- `.ai/proposals/codex-wing-mvp-nowing-frontend-artifact-preservation-20260614.md`
- `Docs/adr/wing-mvp-nowing-frontend-artifact-preservation-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_NOWING_FRONTEND_ARTIFACT_PRESERVATION_20260614.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`

## Alternatives considered

- Change Playwright config output paths globally.
- Ignore the dirty artifact and clean it manually after each run.
- Add a frontend-specific shell wrapper.

## Risks

The restore hook must stay scoped to explicit paths. To reduce risk, the
preserved path list is per-command and repo-boundary validation rejects external
paths.

## Test plan

Run focused no-WING release-check tests, focused WING rails/Director tests,
memory lint, diff check, no-WING quick check, and full pytest.

## Where the other agent may disagree

Kimi may prefer changing Playwright output configuration. I prefer the
release-check preserve hook because it fixes the validation harness directly
and avoids changing frontend test semantics for normal developer workflows.
