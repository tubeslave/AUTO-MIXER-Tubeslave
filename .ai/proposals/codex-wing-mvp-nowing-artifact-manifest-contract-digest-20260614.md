# Proposal: WING MVP no-WING Artifact Manifest Contract Digest

## 1. Short thesis

Add a canonical, self-reference-free digest for the no-WING artifact manifest
contract and surface it everywhere an operator or automation reviews the package.

## 2. What I understood

The active Goal-mode blocker is still `WING read-only connection proof`. Offline
no-WING packets cannot close readiness, but they can make the eventual WING
handoff harder to misuse.

## 3. Proposed solution

Define a canonical manifest contract payload from the required artifact ids,
paths, SHA256 fields, and no-authority booleans. Hash that payload and expose
the digest in `summary.json`, `summary.md`, and `artifact-manifest.json`.

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

- Hash `summary.json` into the summary itself. Rejected because it is
  self-referential.
- Add a separate manifest validation sidecar. Rejected as heavier than the
  current handoff need.

## 6. Risks

Marker drift across the Project Director baseline is the main risk. The digest
must remain canonical and deterministic.

## 7. Test plan

Run focused no-WING tests, Project Director/WING rails tests, full pytest,
`git diff --check`, and memory lint.

## 8. Where the other agent may disagree

Kimi may prefer a full manifest schema validator. The smaller digest is enough
for this slice because the sidecar already carries the generated file hashes.
