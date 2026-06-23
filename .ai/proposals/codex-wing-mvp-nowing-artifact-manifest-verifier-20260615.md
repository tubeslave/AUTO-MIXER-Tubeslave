# Proposal: WING MVP no-WING Artifact Manifest Verifier

## 1. Short thesis

Add a small read-only verifier CLI for no-WING artifact bundles so operators can
validate `artifact-manifest.json` against `summary.json` and `summary.md`
before opening a backend/WING session.

## 2. What I understood

The first open MVP blocker is still WING read-only connection proof. Offline
no-WING packets must not close readiness, but their artifacts need to be
trustworthy enough to hand to an operator before a real read-only WING session.

## 3. Proposed solution

Create `tools/verify_no_wing_artifact_manifest.py`. The tool accepts a
`*_no_wing_release_check` directory, loads `summary.json`, `summary.md`, and
`artifact-manifest.json`, recomputes required artifact SHA256 values, reuses the
existing manifest cross-surface and self-consistency builders, and emits a JSON
report with `status` and `failed_checks`.

## 4. Likely files to touch

- `tools/verify_no_wing_artifact_manifest.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/update_mvp_progress.py`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `Docs/INDEX.md`

## 5. Alternatives considered

- Keep manual runbook inspection only.
- Add verifier logic inside `tools/no_wing_release_check.py`.
- Add another detached artifact.

## 6. Risks

The verifier could drift from the writer if it duplicates contract logic. Reuse
the existing no-WING manifest builders to keep the verifier aligned.

## 7. Test plan

Run focused verifier tests, no-WING quick plus verifier CLI, Project Director
guard checks, memory lint, diff check, and full standard pytest.

## 8. Where the other agent may disagree

Kimi may argue this belongs inside the no-WING release check. Keeping it as a
separate CLI is cleaner for operator preflight because it verifies an already
generated bundle without rewriting artifacts.
