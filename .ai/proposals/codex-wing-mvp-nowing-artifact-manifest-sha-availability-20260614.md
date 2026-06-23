# Proposal: WING MVP no-WING artifact manifest SHA availability

## 1. Short thesis

Add explicit SHA availability and required-count match flags to the no-WING
artifact manifest.

## 2. What I understood

The manifest hashes `summary.json` and `summary.md`, but the contract should
also state that required hashes exist and that the artifact count matches the
required two-file bundle.

## 3. Proposed solution

Add `artifact_manifest_required_artifact_count`,
`artifact_manifest_artifact_count_matches_required`,
`artifact_manifest_required_artifact_sha256_available`,
`artifact_manifest_summary_json_sha256_available`, and
`artifact_manifest_summary_md_sha256_available` to `artifact-manifest.json`.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- WING rail tests and docs

## 5. Alternatives considered

- Infer availability from non-empty SHA strings only.
- Add a separate schema file for the manifest.

## 6. Risks

The main risk is marker drift between the manifest writer, Project Director
baseline, and operator-facing docs. This change does not alter live WING
behavior.

## 7. Test plan

Run focused no-WING manifest tests, Project Director/WING rail tests, full
pytest, `git diff --check`, and memory lint.

## 8. Where the other agent may disagree

Kimi may prefer a stricter manifest schema validator. Inline booleans are the
smallest useful step for the current no-WING handoff bundle.
