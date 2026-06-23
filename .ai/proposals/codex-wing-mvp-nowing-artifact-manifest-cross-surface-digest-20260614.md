# Proposal: WING MVP no-WING Artifact Manifest Cross-Surface Digest

## 1. Short thesis

Add a final sidecar check that binds the manifest contract digest across
`summary.json`, operator-readable `summary.md`, and `artifact-manifest.json`.

## 2. What I understood

The active Goal-mode blocker remains `WING read-only connection proof`. Offline
no-WING artifacts cannot close readiness, but they can reduce evidence drift in
the eventual supervised WING handoff.

## 3. Proposed solution

Expose cross-surface requirement fields in no-WING summary surfaces, then let
`artifact-manifest.json` read those surfaces and record explicit match flags for
contract id, contract SHA256, and expected-match booleans.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`

## 5. Alternatives considered

- Trust the digest fields independently. Rejected because sidecar consumers
  still need a clear aggregate match verdict.
- Hash `artifact-manifest.json` into `summary.json`. Rejected because it would
  create an unstable cycle.

## 6. Risks

The main risk is marker drift across Project Director docs and tests. The
implementation should keep the helper deterministic and avoid changing live
readiness semantics.

## 7. Test plan

Run focused no-WING tests, Project Director/WING rails tests, the full standard
pytest command, no-WING runtime check, `git diff --check`, and memory lint.

## 8. Where the other agent may disagree

Kimi may prefer a broader JSON schema validator for the full sidecar. This
slice keeps scope smaller by only checking the canonical contract digest.
