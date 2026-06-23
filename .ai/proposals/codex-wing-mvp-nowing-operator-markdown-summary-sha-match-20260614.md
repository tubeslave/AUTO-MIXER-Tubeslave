# Proposal: WING MVP no-WING operator markdown summary SHA match

## 1. Short thesis

Add an actual digest and match flag for the no-WING operator markdown summary
binding.

## 2. What I understood

The no-WING handoff bundle now declares a SHA256 for `summary.md`. That is
stronger if the bundle also records the actual digest used for comparison and a
boolean match verdict.

## 3. Proposed solution

Add a helper in `tools/no_wing_release_check.py` that computes the actual
markdown SHA256 from the final markdown text, compares it to the declared
`operator_markdown_summary_sha256`, and fails the summary closed if the declared
value is stale or unavailable.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- Project Director/WING rail tests and docs

## 5. Alternatives considered

- Trust the declared SHA without a match flag.
- Render the actual SHA in markdown.

## 6. Risks

The main risk is stale marker wording in Project Director rails. Rendering the
SHA in markdown remains rejected because it would alter the artifact being
hashed.

## 7. Test plan

Run focused no-WING summary tests, focused Project Director/WING rail tests, full
pytest, `git diff --check`, and memory lint.

## 8. Where the other agent may disagree

Kimi may prefer a separate artifact-manifest validator. This helper is the
smaller step and keeps the integrity verdict in the existing handoff bundle.
