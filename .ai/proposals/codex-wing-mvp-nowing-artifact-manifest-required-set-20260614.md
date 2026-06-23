# Proposal: WING MVP no-WING artifact manifest required set

## 1. Short thesis

Expose the expected artifact set inside the no-WING sidecar manifest.

## 2. What I understood

The manifest currently proves hashes for two files, but a reviewer should also
see that those two files are the expected files, not just any two artifacts.

## 3. Proposed solution

Add `artifact_manifest_required_artifact_ids`,
`artifact_manifest_artifact_ids`, `artifact_manifest_required_artifacts_present`,
and the equivalent path fields to `artifact-manifest.json`.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- WING rail tests and docs

## 5. Alternatives considered

- Keep only `artifact_manifest_artifact_count`.
- Add a separate manifest schema file.

## 6. Risks

The main risk is stale wording in Project Director markers. The implementation
does not change live WING behavior.

## 7. Test plan

Run focused no-WING summary tests, Project Director/WING rail tests, full pytest,
`git diff --check`, and memory lint.

## 8. Where the other agent may disagree

Kimi may prefer a separate schema file. Inline manifest fields are smaller for
this no-WING handoff slice.
