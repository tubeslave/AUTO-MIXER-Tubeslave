# Codex Proposal: WING MVP Current Command Markdown Summary Surface

## 1. Short thesis

Bind the operator-readable no-WING `summary.md` proof-plan section into the
Project Director baseline rail, so supervised proof handoff cannot rely only on
machine-readable summary fields.

## 2. What I understood

The previous slice made the no-WING markdown report render the required and
missing supervised proof-plan summary fields. The remaining gap is that Project
Director baseline coverage does not yet name the markdown surface itself as a
required operator-facing artifact.

## 3. Proposed solution

Add an explicit `operator-readable no-WING summary.md` marker to the Current
First Command baseline list. Mirror that language in the Current First Command,
Codex cycle prompt, README, and tests.

## 4. Likely files to touch

- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/adr/wing-mvp-current-command-markdown-summary-surface-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_COMMAND_MARKDOWN_SUMMARY_SURFACE_20260614.md`

## 5. Alternatives considered

- Only rely on `tools/no_wing_release_check.py` tests. Rejected because Project
  Director would not enforce the operator-readable handoff surface.
- Add another schema field. Rejected because the existing schema already carries
  the data; the gap is visibility in `summary.md`.

## 6. Risks

The main risk is marker drift between Current First Command, prompt, README, and
tests. Keeping the marker short and tested reduces that risk.

## 7. Test plan

Run focused Project Director / WING rail tests, `git diff --check`, and memory
lint. This change does not need WING hardware.

## 8. Where the other agent may disagree

Kimi may argue the marker is redundant because markdown rendering is already
tested. The counterpoint is that Goal-mode task selection needs the same
operator-facing constraint before supervised proof planning.
